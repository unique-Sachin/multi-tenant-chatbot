"""Retrieval system for Zibtek chatbot using LangChain and Pinecone."""

import os
from typing import List, Optional, Dict, Any
from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever
from pinecone import Pinecone

load_dotenv()


class ZibtekRetriever:
    """Retrieval system for Zibtek knowledge base."""
    
    def __init__(self):
        """Initialize the retriever with Pinecone and OpenAI."""
        # Get environment variables
        self.pinecone_api_key = os.getenv("PINECONE_API_KEY")
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        
        if not self.pinecone_api_key:
            raise ValueError("PINECONE_API_KEY environment variable is required")
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        
        # Initialize Pinecone
        self.pc = Pinecone(api_key=self.pinecone_api_key)
        self.index_name = "zibtek-chatbot-index"
        
        # Initialize OpenAI embeddings
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small"
        )
        
        # Initialize vector store (namespace will be specified per query)
        self.vector_store = PineconeVectorStore(
            index=self.pc.Index(self.index_name),
            embedding=self.embeddings
        )
        
        print("✅ Zibtek Retriever initialized")
        print(f"   - Index: {self.index_name}")
        print(f"   - Namespace: dynamic (per query)")
        print(f"   - Embedding model: text-embedding-3-small")
    
    def make_pinecone_retriever(
        self,
        k: int = 5,
        namespace: str = "zibtek",
        score_threshold: Optional[float] = None
    ) -> VectorStoreRetriever:
        """Create a LangChain retriever for Pinecone.
        
        Args:
            k: Number of documents to retrieve
            namespace: Pinecone namespace to search
            score_threshold: Minimum similarity score threshold
            
        Returns:
            VectorStoreRetriever: Configured LangChain retriever
        """
        # Create search kwargs
        search_kwargs = {
            "k": k,
            "namespace": namespace
        }
        
        if score_threshold is not None:
            search_kwargs["score_threshold"] = score_threshold
        
        # Create retriever
        retriever = self.vector_store.as_retriever(
            search_type="similarity_score_threshold" if score_threshold else "similarity",
            search_kwargs=search_kwargs
        )
        
        return retriever
    
    def retrieve(
        self,
        query: str,
        k: int = 5,
        namespace: str = "zibtek",
        score_threshold: Optional[float] = None
    ) -> List[Document]:
        """Retrieve documents for a query.
        
        Args:
            query: Search query
            k: Number of documents to retrieve
            namespace: Pinecone namespace to search
            score_threshold: Minimum similarity score threshold
            
        Returns:
            List[Document]: Retrieved documents with metadata including scores
        """
        try:
            # Create retriever
            retriever = self.make_pinecone_retriever(
                k=k,
                namespace=namespace,
                score_threshold=score_threshold
            )
            
            # Perform retrieval
            docs = retriever.invoke(query)
            
            # Add scores to metadata if available
            # Note: LangChain's similarity_score_threshold search includes scores
            for doc in docs:
                if hasattr(doc, 'metadata') and 'score' not in doc.metadata:
                    # For compatibility, we'll perform a direct similarity search
                    # to get scores when using regular similarity search
                    pass
            
            return docs
            
        except Exception as e:
            print(f"❌ Retrieval error: {e}")
            return []
    
    def retrieve_with_scores(
        self,
        query: str,
        k: int = 5,
        namespace: str = "zibtek",
        score_threshold: Optional[float] = None
    ) -> List[tuple[Document, float]]:
        """Retrieve documents with explicit scores.
        
        Args:
            query: Search query
            k: Number of documents to retrieve
            namespace: Pinecone namespace to search
            score_threshold: Minimum similarity score threshold
            
        Returns:
            List[tuple[Document, float]]: Documents with their similarity scores
        """
        try:
            # Use direct similarity search with scores
            search_kwargs = {
                "k": k,
                "namespace": namespace
            }
            
            docs_with_scores = self.vector_store.similarity_search_with_score(
                query,
                **search_kwargs
            )
            
            # Filter by score threshold if provided
            if score_threshold is not None:
                docs_with_scores = [
                    (doc, score) for doc, score in docs_with_scores
                    if score >= score_threshold
                ]
            
            # Add scores to document metadata
            for doc, score in docs_with_scores:
                doc.metadata["score"] = score
            
            return docs_with_scores
            
        except Exception as e:
            print(f"❌ Retrieval with scores error: {e}")
            return []
    
    def search_by_metadata(
        self,
        query: str,
        metadata_filter: Dict[str, Any],
        k: int = 5,
        namespace: str = "zibtek"
    ) -> List[Document]:
        """Search with custom metadata filters.
        
        Args:
            query: Search query
            metadata_filter: Additional metadata filters
            k: Number of documents to retrieve
            namespace: Pinecone namespace to search
            
        Returns:
            List[Document]: Filtered and retrieved documents
        """
        try:
            # Combine default filter with custom filters
            combined_filter = {"site": "https://www.zibtek.com"}
            combined_filter.update(metadata_filter)
            
            docs = self.vector_store.similarity_search(
                query,
                k=k,
                namespace=namespace,
                filter=combined_filter
            )
            
            return docs
            
        except Exception as e:
            print(f"❌ Metadata search error: {e}")
            return []
    
    def get_retriever_stats(self, namespace: str = "zibtek") -> Dict[str, Any]:
        """Get statistics about the retriever index.
        
        Args:
            namespace: Pinecone namespace to check
            
        Returns:
            Dict[str, Any]: Index statistics
        """
        try:
            index = self.pc.Index(self.index_name)
            stats = index.describe_index_stats()
            
            namespace_stats = stats.get('namespaces', {}).get(namespace, {})
            
            return {
                "total_vectors": namespace_stats.get('vector_count', 0),
                "dimension": stats.get('dimension', 0),
                "index_fullness": stats.get('index_fullness', 0),
                "namespace": namespace
            }
            
        except Exception as e:
            print(f"❌ Stats error: {e}")
            return {}


def make_pinecone_retriever(
    k: int = 5,
    namespace: str = "zibtek",
    score_threshold: Optional[float] = None
) -> VectorStoreRetriever:
    """Convenience function to create a Pinecone retriever.
    
    Args:
        k: Number of documents to retrieve
        namespace: Pinecone namespace to search
        score_threshold: Minimum similarity score threshold
        
    Returns:
        VectorStoreRetriever: Configured LangChain retriever
    """
    retriever_instance = ZibtekRetriever()
    return retriever_instance.make_pinecone_retriever(
        k=k,
        namespace=namespace,
        score_threshold=score_threshold
    )


def retrieve(query: str, k: int = 5) -> List[Document]:
    """Convenience function for simple retrieval.
    
    Args:
        query: Search query
        k: Number of documents to retrieve
        
    Returns:
        List[Document]: Retrieved documents with metadata
    """
    retriever_instance = ZibtekRetriever()
    return retriever_instance.retrieve(query=query, k=k)


# Example usage and testing
if __name__ == "__main__":
    # Test the retriever
    try:
        retriever = ZibtekRetriever()
        
        # Test query
        test_query = "What services does Zibtek offer?"
        print(f"\n🔍 Testing retrieval for: '{test_query}'")
        
        # Retrieve documents
        docs = retriever.retrieve(test_query, k=3)
        
        print(f"✅ Retrieved {len(docs)} documents")
        
        for i, doc in enumerate(docs, 1):
            print(f"\n📄 Document {i}:")
            print(f"   URL: {doc.metadata.get('url', 'N/A')}")
            print(f"   Title: {doc.metadata.get('title', 'N/A')}")
            print(f"   Score: {doc.metadata.get('score', 'N/A')}")
            print(f"   Content: {doc.page_content[:200]}...")
        
        # Get stats
        stats = retriever.get_retriever_stats()
        print(f"\n📊 Index Stats: {stats}")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")