"""Hybrid retrieval system combining Pinecone vector search with BM25 keyword search.

This module implements:
1. BM25 index for exact term matching and keyword recall
2. Reciprocal Rank Fusion (RRF) to merge vector and keyword results
3. Redis caching for the BM25 index to improve startup time
4. Fallback mechanisms for robust operation
"""

import os
import json
import pickle
import logging
from typing import List, Tuple, Dict, Optional, Any
from dataclasses import dataclass
from collections import defaultdict

import redis
from rank_bm25 import BM25Okapi
from langchain.schema import Document
from langchain_pinecone import PineconeVectorStore

# Setup logging
logger = logging.getLogger(__name__)


@dataclass
class HybridResult:
    """Result from hybrid retrieval."""
    documents: List[Document]
    vector_scores: Dict[str, float]
    bm25_scores: Dict[str, float] 
    fusion_scores: Dict[str, float]
    method: str  # 'hybrid', 'vector_only', 'bm25_only'


class BM25Index:
    """Lightweight BM25 index for keyword-based retrieval."""
    
    def __init__(self):
        self.index: Optional[BM25Okapi] = None
        self.doc_ids: List[str] = []
        self.doc_texts: List[str] = []
        self.doc_metadata: Dict[str, Dict] = {}
        self.is_built = False
    
    def build_from_documents(self, documents: List[Document]) -> None:
        """Build BM25 index from document list."""
        logger.info(f"🔍 Building BM25 index from {len(documents)} documents...")
        
        self.doc_ids = []
        self.doc_texts = []
        self.doc_metadata = {}
        
        # Prepare documents for BM25
        tokenized_docs = []
        for doc in documents:
            doc_id = doc.metadata.get('id', f"doc_{len(self.doc_ids)}")
            text = doc.page_content
            
            self.doc_ids.append(doc_id)
            self.doc_texts.append(text)
            self.doc_metadata[doc_id] = doc.metadata
            
            # Simple tokenization for BM25
            tokens = text.lower().split()
            tokenized_docs.append(tokens)
        
        # Build BM25 index
        self.index = BM25Okapi(tokenized_docs)
        self.is_built = True
        
        logger.info(f"✅ BM25 index built with {len(self.doc_ids)} documents")
    
    def search(self, query: str, k: int = 20) -> List[Tuple[str, float]]:
        """Search BM25 index and return (doc_id, score) tuples."""
        if not self.is_built or self.index is None:
            logger.warning("⚠️  BM25 index not built, returning empty results")
            return []
        
        # Tokenize query
        query_tokens = query.lower().split()
        
        # Get BM25 scores
        scores = self.index.get_scores(query_tokens)
        
        # Create results with doc_ids and scores
        results = [(self.doc_ids[i], float(score)) for i, score in enumerate(scores)]
        
        # Sort by score descending and take top k
        results.sort(key=lambda x: x[1], reverse=True)
        
        return results[:k]
    
    def get_document_by_id(self, doc_id: str) -> Optional[Document]:
        """Retrieve document by ID."""
        if doc_id not in self.doc_metadata:
            return None
        
        doc_index = self.doc_ids.index(doc_id)
        return Document(
            page_content=self.doc_texts[doc_index],
            metadata=self.doc_metadata[doc_id]
        )
    
    def to_cache_data(self) -> Dict[str, Any]:
        """Serialize index data for caching."""
        if not self.is_built or self.index is None:
            return {}
        
        return {
            'doc_ids': self.doc_ids,
            'doc_texts': self.doc_texts,
            'doc_metadata': self.doc_metadata,
            'index_data': {
                'corpus_size': self.index.corpus_size,
                'avgdl': self.index.avgdl,
                'doc_freqs': self.index.doc_freqs,
                'idf': self.index.idf,
                'doc_len': self.index.doc_len
            }
        }
    
    def from_cache_data(self, data: Dict[str, Any]) -> bool:
        """Load index from cached data."""
        try:
            self.doc_ids = data['doc_ids']
            self.doc_texts = data['doc_texts'] 
            self.doc_metadata = data['doc_metadata']
            
            # Rebuild tokenized corpus
            tokenized_docs = []
            for text in self.doc_texts:
                tokens = text.lower().split()
                tokenized_docs.append(tokens)
            
            # Create BM25 index
            self.index = BM25Okapi(tokenized_docs)
            
            # Restore BM25 internal state
            index_data = data['index_data']
            self.index.corpus_size = index_data['corpus_size']
            self.index.avgdl = index_data['avgdl']
            self.index.doc_freqs = index_data['doc_freqs']
            self.index.idf = index_data['idf']
            self.index.doc_len = index_data['doc_len']
            
            self.is_built = True
            logger.info(f"✅ BM25 index loaded from cache with {len(self.doc_ids)} documents")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to load BM25 index from cache: {e}")
            return False


class HybridRetriever:
    """Hybrid retrieval system combining vector and keyword search."""
    
    def __init__(self, vector_store: PineconeVectorStore, namespace: str = "zibtek"):
        """Initialize hybrid retriever.
        
        NOTE: BM25 index is built for the specified namespace at initialization.
        For multi-tenant use, the vector component will use the dynamic namespace parameter,
        but BM25 will always use the init namespace. For full multi-tenant BM25 support,
        consider disabling hybrid search or rebuilding the index per namespace.
        """
        self.vector_store = vector_store
        self.namespace = namespace
        self.bm25 = BM25Index()
        self.redis_client: Optional[redis.Redis] = None
        self.cache_key = f"bm25_index_v1_{namespace}"  # Namespace-specific cache key
        self.max_docs_for_bm25 = 10000
        
        # Initialize Redis if available
        self._init_redis()
        
        # Build or load BM25 index for this namespace
        self._initialize_bm25_index()
    
    def _init_redis(self) -> None:
        """Initialize Redis client if available."""
        try:
            redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379/0')
            self.redis_client = redis.from_url(redis_url, decode_responses=False)
            # Test connection
            self.redis_client.ping()
            logger.info("✅ Redis client initialized successfully")
        except Exception as e:
            logger.warning(f"⚠️  Redis not available: {e}")
            self.redis_client = None
    
    def _load_bm25_from_cache(self) -> bool:
        """Load BM25 index from Redis cache."""
        if not self.redis_client:
            return False
        
        try:
            logger.info("🔍 Checking Redis cache for BM25 index...")
            cached_data_bytes = self.redis_client.get(self.cache_key)
            
            if cached_data_bytes and isinstance(cached_data_bytes, bytes):
                data = pickle.loads(cached_data_bytes)
                success = self.bm25.from_cache_data(data)
                if success:
                    logger.info("✅ BM25 index loaded from Redis cache")
                    return True
            
            logger.info("📭 No valid BM25 cache found in Redis")
            return False
            
        except Exception as e:
            logger.warning(f"⚠️  Failed to load BM25 cache from Redis: {e}")
            return False
    
    def _save_bm25_to_cache(self) -> None:
        """Save BM25 index to Redis cache."""
        if not self.redis_client or not self.bm25.is_built:
            return
        
        try:
            data = self.bm25.to_cache_data()
            cached_data = pickle.dumps(data)
            
            # Set with 7 day expiration
            self.redis_client.setex(self.cache_key, 7 * 24 * 3600, cached_data)
            logger.info("✅ BM25 index saved to Redis cache")
            
        except Exception as e:
            logger.warning(f"⚠️  Failed to save BM25 cache to Redis: {e}")
    
    def _fetch_documents_from_pinecone(self) -> List[Document]:
        """Fetch documents from Pinecone for BM25 indexing."""
        logger.info(f"📥 Fetching documents from Pinecone (limit: {self.max_docs_for_bm25})...")
        
        try:
            # Use similarity search with a very broad query to get diverse results
            query = "information data content"
            docs = self.vector_store.similarity_search(
                query=query,
                k=min(self.max_docs_for_bm25, 1000),
                namespace=self.namespace
            )
            
            # If we need more docs, try additional queries
            if len(docs) < self.max_docs_for_bm25:
                additional_queries = [
                    "development software technology",
                    "services solutions business",
                    "company about team"
                ]
                
                seen_ids = set()
                for doc in docs:
                    doc_id = doc.metadata.get('id', doc.page_content[:50])
                    seen_ids.add(doc_id)
                
                for additional_query in additional_queries:
                    if len(docs) >= self.max_docs_for_bm25:
                        break
                    
                    additional_docs = self.vector_store.similarity_search(
                        query=additional_query,
                        k=min(500, self.max_docs_for_bm25 - len(docs)),
                        namespace=self.namespace
                    )
                    
                    # Deduplicate
                    for doc in additional_docs:
                        doc_id = doc.metadata.get('id', doc.page_content[:50])
                        if doc_id not in seen_ids:
                            docs.append(doc)
                            seen_ids.add(doc_id)
            
            logger.info(f"✅ Fetched {len(docs)} documents from Pinecone")
            return docs
            
        except Exception as e:
            logger.error(f"❌ Failed to fetch documents from Pinecone: {e}")
            return []
    
    def _initialize_bm25_index(self) -> None:
        """Initialize BM25 index, loading from cache or building fresh."""
        logger.info("🚀 Initializing BM25 index...")
        
        # Try to load from cache first
        if self._load_bm25_from_cache():
            return
        
        # Build fresh index
        documents = self._fetch_documents_from_pinecone()
        if documents:
            self.bm25.build_from_documents(documents)
            self._save_bm25_to_cache()
        else:
            logger.warning("⚠️  No documents available for BM25 indexing")
    
    def bm25_search(self, query: str, k: int = 20) -> List[Tuple[str, float]]:
        """Perform BM25 keyword search."""
        return self.bm25.search(query, k)
    
    def reciprocal_rank_fusion(
        self,
        vector_results: List[Tuple[str, float]],
        bm25_results: List[Tuple[str, float]],
        k: int = 60
    ) -> List[Tuple[str, float]]:
        """
        Combine results using Reciprocal Rank Fusion.
        
        RRF formula: score = Σ(1 / (rank + k)) for each ranking
        Common k value is 60.
        """
        fusion_scores = defaultdict(float)
        
        # Add vector search scores
        for rank, (doc_id, score) in enumerate(vector_results):
            fusion_scores[doc_id] += 1.0 / (rank + k)
        
        # Add BM25 scores  
        for rank, (doc_id, score) in enumerate(bm25_results):
            fusion_scores[doc_id] += 1.0 / (rank + k)
        
        # Sort by fusion score
        fused_results = [(doc_id, score) for doc_id, score in fusion_scores.items()]
        fused_results.sort(key=lambda x: x[1], reverse=True)
        
        return fused_results
    
    def hybrid_search(
        self,
        query: str,
        k: int = 20,
        vector_k: int = 50,
        bm25_k: int = 50,
        rrf_k: int = 60,
        namespace: Optional[str] = None
    ) -> HybridResult:
        """
        Perform hybrid search combining vector and BM25 results.
        
        Args:
            query: Search query
            k: Final number of documents to return
            vector_k: Number of results from vector search
            bm25_k: Number of results from BM25 search
            rrf_k: RRF parameter (typically 60)
            namespace: Pinecone namespace to search (overrides self.namespace if provided)
        """
        # Use provided namespace or fall back to instance namespace
        search_namespace = namespace if namespace is not None else self.namespace
        logger.info(f"🔍 Performing hybrid search for: '{query[:50]}...' in namespace: '{search_namespace}'")
        
        # Get vector search results
        try:
            vector_docs = self.vector_store.similarity_search_with_score(
                query=query,
                k=vector_k,
                namespace=search_namespace
            )
            vector_results = []
            vector_scores = {}
            
            for doc, score in vector_docs:
                doc_id = doc.metadata.get('id', f"vec_{len(vector_results)}")
                vector_results.append((doc_id, float(score)))
                vector_scores[doc_id] = float(score)
            
            logger.info(f"✅ Vector search: {len(vector_results)} results")
            
        except Exception as e:
            logger.error(f"❌ Vector search failed: {e}")
            vector_results = []
            vector_scores = {}
        
        # Get BM25 search results
        try:
            bm25_results = self.bm25_search(query, bm25_k)
            bm25_scores = dict(bm25_results)
            logger.info(f"✅ BM25 search: {len(bm25_results)} results")
            
        except Exception as e:
            logger.error(f"❌ BM25 search failed: {e}")
            bm25_results = []
            bm25_scores = {}
        
        # Determine search method
        if not vector_results and not bm25_results:
            logger.warning("⚠️  Both vector and BM25 search failed")
            return HybridResult([], {}, {}, {}, "failed")
        
        if not bm25_results:
            logger.info("📊 Using vector-only results (BM25 unavailable)")
            method = "vector_only"
            fusion_results = vector_results[:k]
            fusion_scores = {doc_id: score for doc_id, score in fusion_results}
        elif not vector_results:
            logger.info("📊 Using BM25-only results (vector search unavailable)")
            method = "bm25_only"
            fusion_results = bm25_results[:k]
            fusion_scores = {doc_id: score for doc_id, score in fusion_results}
        else:
            logger.info("🔀 Fusing vector and BM25 results with RRF")
            method = "hybrid"
            fusion_results = self.reciprocal_rank_fusion(
                vector_results, bm25_results, rrf_k
            )
            fusion_scores = {doc_id: score for doc_id, score in fusion_results}
        
        # Retrieve final documents
        final_docs = []
        final_fusion_scores = {}
        
        for doc_id, fusion_score in fusion_results[:k]:
            # Try to get document from BM25 index first
            doc = self.bm25.get_document_by_id(doc_id)
            
            # If not found, try vector store
            if doc is None:
                try:
                    # Search by metadata id
                    docs = self.vector_store.similarity_search(
                        query=query,
                        k=100,
                        namespace=self.namespace,
                        filter={"id": doc_id}
                    )
                    if docs:
                        doc = docs[0]
                except:
                    pass
            
            if doc:
                final_docs.append(doc)
                final_fusion_scores[doc_id] = fusion_score
        
        logger.info(f"🎯 Hybrid search complete: {len(final_docs)} documents, method: {method}")
        
        return HybridResult(
            documents=final_docs,
            vector_scores=vector_scores,
            bm25_scores=bm25_scores,
            fusion_scores=final_fusion_scores,
            method=method
        )


# Test function
if __name__ == "__main__":
    import sys
    sys.path.append('/Users/sachinmishra/Documents/zibtek-assgn')
    
    from src.retrieval.retriever import ZibtekRetriever
    
    # Initialize retriever 
    print("🧪 Testing HybridRetriever...")
    
    try:
        # Get Pinecone vector store from existing retriever
        retriever = ZibtekRetriever()
        vector_store = retriever.vector_store
        
        # Initialize hybrid retriever
        hybrid = HybridRetriever(vector_store)
        
        # Test searches
        test_queries = [
            "What programming languages does Zibtek use?",
            "PHP Laravel development services",
            "Android app development"
        ]
        
        for query in test_queries:
            print(f"\n🔍 Testing: {query}")
            result = hybrid.hybrid_search(query, k=5)
            
            print(f"📊 Method: {result.method}")
            print(f"📄 Documents: {len(result.documents)}")
            print(f"🎯 Vector scores: {len(result.vector_scores)}")
            print(f"🔤 BM25 scores: {len(result.bm25_scores)}")
            print(f"🔀 Fusion scores: {len(result.fusion_scores)}")
            
            if result.documents:
                print("Top documents:")
                for i, doc in enumerate(result.documents[:3], 1):
                    content = doc.page_content[:100] + "..."
                    print(f"  {i}. {content}")
        
        print("\n✅ HybridRetriever test completed!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()