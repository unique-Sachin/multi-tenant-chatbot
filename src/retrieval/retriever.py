"""Milvus-based retrieval system for Zibtek chatbot.

Uses Milvus/Zilliz Cloud with hybrid search (dense vectors + BM25 sparse vectors).
"""

import os
from typing import List, Optional, Dict, Any, Tuple
from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from pymilvus import MilvusClient, AnnSearchRequest, RRFRanker


load_dotenv()

class MilvusRetriever:
    """Milvus-based retriever with built-in hybrid search (dense + BM25)."""
    
    def __init__(self):
        """Initialize Milvus retriever."""
        self.milvus_uri = os.getenv("MILVUS_URI")
        self.milvus_token = os.getenv("MILVUS_TOKEN")
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.collection_name = os.getenv("MILVUS_COLLECTION", "documents")
        
        if not self.milvus_uri or not self.milvus_token:
            raise ValueError("MILVUS_URI and MILVUS_TOKEN environment variables are required")
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        
        # Initialize Milvus client
        self.client = MilvusClient(uri=self.milvus_uri, token=self.milvus_token)
        
        # Initialize OpenAI embeddings
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        
        print("✅ Milvus Retriever initialized")
        print(f"   - Collection: {self.collection_name}")
        print(f"   - Embedding model: text-embedding-3-small")
        print(f"   - Hybrid search: Dense + BM25 with RRF")
    
    def hybrid_search(
        self,
        query: str,
        k: int = 5,
        partition_name: str = "_default",
    ) -> Any:
        """Perform hybrid search (dense + BM25) with multi-tenant support.
        
        Args:
            query: Search query text
            k: Number of results to return
            partition_name: Partition name for tenant isolation (e.g., "org_zibtek")
            dense_weight: Weight for dense vector search (0-1)
            sparse_weight: Weight for sparse BM25 search (0-1)
        
        Returns:
            List of (Document, score) tuples
        """
        try:
            # Generate query embedding
            query_vector = self.embeddings.embed_query(query)
            
            # Load collection if not loaded
            try:
                self.client.load_collection(self.collection_name)
            except:
                pass  # Already loaded
            
            # Create search requests
            # Use larger multiplier for better RRF results
            # RRF needs diverse candidate pools to effectively combine rankings
            search_limit = max(k * 2, 20)  # At least 20 candidates, or 2x requested results
            
            dense_req = AnnSearchRequest(
                data=[query_vector],
                anns_field="dense_vector",
                param={"metric_type": "COSINE"},
                limit=search_limit
            )

            
            sparse_req = AnnSearchRequest(
                data=[query],  # Milvus will tokenize and search BM25
                anns_field="sparse_vector",
                param={"metric_type": "BM25"},
                limit=search_limit
            )
            
            # Perform hybrid search
            results = self.client.hybrid_search(
                collection_name=self.collection_name,
                reqs=[dense_req, sparse_req],
                ranker=RRFRanker(),  # Reciprocal Rank Fusion
                limit=k,
                partition_names=[partition_name],  # Tenant isolation
                output_fields=["text", "id", "url", "title", "description", "chunk_index"]
            )
            
            # Convert to LangChain Document format
            documents = []
            for hits in results:
                for hit in hits:
                    entity = hit.get('entity', {})
                    
                    # Use text for display
                    display_text = entity.get('text', '')
                    
                    # Create Document with metadata
                    doc = Document(
                        page_content=display_text,
                        metadata={
                            'id': entity.get('id', ''),
                            'url': entity.get('url', ''),
                            'title': entity.get('title', ''),
                            'description': entity.get('description', ''),
                            'chunk_index': entity.get('chunk_index', 0),
                            'score': hit.get('distance', 0.0),
                            'partition': partition_name,
                        }
                    )
                    documents.append((doc, hit.get('distance', 0.0)))
            
            return documents
            
        except Exception as e:
            print(f"❌ Hybrid search error: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def retrieve_with_scores(
        self,
        query: str,
        k: int = 5,
        partition_name: str = "_default"
    ) -> List[Tuple[Document, float]]:
        """Alias for hybrid_search to maintain compatibility with existing code.
        
        Args:
            query: Search query
            k: Number of results
            partition_name: Partition for tenant isolation (replaces namespace)
        
        Returns:
            List of (Document, score) tuples
        """
        return self.hybrid_search(query, k, partition_name)
    
    def search_by_url(
        self,
        url: str,
        partition_name: str = "_default",
        exact_match: bool = True
    ) -> List[Document]:
        """Search documents by URL metadata.
        
        Args:
            url: URL to search for
            partition_name: Partition name for tenant isolation
            exact_match: If True, match exact URL. If False, match URLs containing the string.
        
        Returns:
            List of Document objects matching the URL
        """
        try:
            # Load collection if not loaded
            try:
                self.client.load_collection(self.collection_name)
            except:
                pass  # Already loaded
            
            # Build filter expression
            if exact_match:
                filter_expr = f'url == "{url}"'
            else:
                # For partial match, use LIKE operator
                filter_expr = f'url like "%{url}%"'
            
            # Query by filter
            results = self.client.query(
                collection_name=self.collection_name,
                filter=filter_expr,
                output_fields=["text", "id", "url", "title", "description", "chunk_index"],
                partition_names=[partition_name]
            )
            
            # Convert to LangChain Document format
            documents = []
            for entity in results:
                # Use text for display
                display_text = entity.get('text', '')
                
                doc = Document(
                    page_content=display_text,
                    metadata={
                        'id': entity.get('id', ''),
                        'url': entity.get('url', ''),
                        'title': entity.get('title', ''),
                        'description': entity.get('description', ''),
                        'chunk_index': entity.get('chunk_index', 0),
                        'partition': partition_name,
                    }
                )
                documents.append(doc)
            
            return documents
            
        except Exception as e:
            print(f"❌ URL search error: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def get_retriever_stats(self, partition_name: str = "_default") -> Dict[str, Any]:
        """Get collection/partition statistics.
        
        Args:
            partition_name: Partition to check stats for
        
        Returns:
            Dictionary with statistics
        """
        try:
            stats = self.client.describe_collection(self.collection_name)
            partitions = self.client.list_partitions(self.collection_name)
            
            return {
                "collection_name": self.collection_name,
                "total_entities": stats.get("num_entities", 0),
                "partitions": partitions,
                "partition": partition_name,
                "dimension": 1536,
                "search_type": "hybrid (dense + BM25 + RRF)"
            }
            
        except Exception as e:
            print(f"❌ Stats error: {e}")
            return {}


# Example usage
if __name__ == "__main__":
    print("=" * 60)
    print("Testing Milvus Retriever")
    print("=" * 60)
    
    try:
        retriever = MilvusRetriever()
        
        # Test query
        test_query = "Is there any phone number to contact Zibtek?"
        partition = "zibtek"
        
        print(f"\n🔍 Testing hybrid search")
        print(f"   Query: '{test_query}'")
        print(f"   Partition: '{partition}'")
        
        # Retrieve documents
        docs = retriever.hybrid_search(test_query, k=3, partition_name=partition)
        
        # print(f"\n✅ Retrieved {len(docs)} documents")x
        # print(docs)
        
        for i, (doc, score) in enumerate(docs, 1):
            print(f"\n📄 Document {i}:")
            print(f"   ID: {doc.metadata.get('id', 'N/A')}")
            print(f"   URL: {doc.metadata.get('url', 'N/A')}")
            print(f"   Score: {score:.4f}")
            print(f"   Content: {doc.page_content}...")
        
        # Get stats
        stats = retriever.get_retriever_stats(partition)
        print(f"\n📊 Collection Stats:")
        for key, value in stats.items():
            print(f"   {key}: {value}")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
