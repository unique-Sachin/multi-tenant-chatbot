"""Pinecone storage helpers for the Zibtek chatbot."""

import os
import time
from typing import List, Dict, Any, Optional, Tuple
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

load_dotenv()


class PineconeStorage:
    """Pinecone vector storage client."""
    
    def __init__(self):
        """Initialize Pinecone client and index."""
        self.api_key = os.getenv("PINECONE_API_KEY")
        self.index_name = os.getenv("PINECONE_INDEX", "zibtek-chatbot-index")
        self.namespace = "zibtek"
        
        if not self.api_key:
            print("⚠️  PINECONE_API_KEY not found. Pinecone operations will be disabled.")
            self.pc = None
            self.index = None
            return
        
        # Initialize Pinecone client
        self.pc = Pinecone(api_key=self.api_key)
        self.index = None
        
        print(f"✅ Pinecone client initialized")
        print(f"   - Index: {self.index_name}")
        print(f"   - Namespace: {self.namespace}")
    
    def init_index(self, dimension: int = 1536, create_if_not_exists: bool = True) -> bool:
        """Initialize or connect to Pinecone index."""
        if not self.pc:
            print("❌ Pinecone client not initialized")
            return False
            
        try:
            # Check if index exists
            existing_indexes = [index.name for index in self.pc.list_indexes()]
            
            if self.index_name in existing_indexes:
                print(f"✅ Index '{self.index_name}' already exists")
                self.index = self.pc.Index(self.index_name)
                return True
            
            elif create_if_not_exists:
                print(f"🔧 Creating new index '{self.index_name}'...")
                
                # Create index with cosine similarity (serverless)
                self.pc.create_index(
                    name=self.index_name,
                    dimension=dimension,
                    metric="cosine",
                    spec=ServerlessSpec(
                        cloud="aws",
                        region="us-east-1"
                    )
                )
                
                # Wait for index to be ready
                print("⏳ Waiting for index to be ready...")
                attempts = 0
                max_attempts = 30
                
                while attempts < max_attempts:
                    try:
                        self.index = self.pc.Index(self.index_name)
                        stats = self.index.describe_index_stats()
                        print(f"✅ Index created successfully: dimension={stats.dimension}")
                        return True
                    except Exception as e:
                        print(f"   Still creating... (attempt {attempts + 1}/{max_attempts})")
                        time.sleep(10)
                        attempts += 1
                
                print("❌ Timeout waiting for index to be ready")
                return False
            
            else:
                print(f"❌ Index '{self.index_name}' not found and create_if_not_exists=False")
                return False
                
        except Exception as e:
            print(f"❌ Error initializing index: {e}")
            return False
    
    def upsert_chunks(
        self, 
        chunks: List[Dict[str, Any]], 
        batch_size: int = 100,
        namespace: Optional[str] = None
    ) -> bool:
        """Upsert chunks to Pinecone index under specified namespace."""
        if not self.index:
            print("❌ Index not initialized. Call init_index() first.")
            return False
        
        # Use provided namespace or default
        target_namespace = namespace or self.namespace
        
        try:
            total_chunks = len(chunks)
            print(f"📤 Upserting {total_chunks} chunks to Pinecone (namespace: {target_namespace})...")
            
            # Process in batches
            for i in range(0, total_chunks, batch_size):
                batch = chunks[i:i + batch_size]
                
                # Prepare vectors for upsert
                vectors = []
                for chunk in batch:
                    vector = {
                        "id": chunk["id"],
                        "values": chunk["embedding"],
                        "metadata": {
                            "text": chunk["text"][:40000],  # Pinecone metadata limit
                            "url": chunk["metadata"]["url"],
                            "title": chunk["metadata"].get("title", ""),
                            "section": chunk["metadata"].get("section", ""),
                            "crawl_ts": chunk["metadata"]["crawl_ts"],
                            "site": chunk["metadata"]["site"],
                            "namespace": chunk["metadata"].get("namespace", target_namespace),
                            "chunk_index": chunk["metadata"].get("chunk_index", 0),
                        }
                    }
                    vectors.append(vector)
                
                # Upsert batch
                self.index.upsert(
                    vectors=vectors,
                    namespace=target_namespace
                )
                
                print(f"   ✅ Upserted batch {i//batch_size + 1}/{(total_chunks-1)//batch_size + 1}")
                
                # Small delay to avoid rate limits
                time.sleep(0.1)
            
            print(f"✅ Successfully upserted all {total_chunks} chunks")
            return True
            
        except Exception as e:
            print(f"❌ Error upserting chunks: {e}")
            return False
    
    def query_embeddings(
        self, 
        query_embedding: List[float], 
        top_k: int = 10,
        filter_dict: Optional[Dict[str, Any]] = None,
        include_metadata: bool = True
    ) -> List[Dict[str, Any]]:
        """Query Pinecone index for similar embeddings."""
        if not self.index:
            print("❌ Index not initialized. Call init_index() first.")
            return []
        
        try:
            response = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                namespace=self.namespace,
                filter=filter_dict,
                include_metadata=include_metadata,
                include_values=False
            )
            
            results = []
            # Handle response properly
            if hasattr(response, 'matches') and response.matches:  # type: ignore
                for match in response.matches:  # type: ignore
                    result = {
                        "id": match.id,
                        "score": float(match.score),
                        "metadata": match.metadata if include_metadata else None
                    }
                    results.append(result)
            
            return results
            
        except Exception as e:
            print(f"❌ Error querying embeddings: {e}")
            return []
    
    def get_index_stats(self) -> Dict[str, Any]:
        """Get statistics about the Pinecone index."""
        if not self.index:
            return {"error": "Index not initialized"}
        
        try:
            stats = self.index.describe_index_stats()
            return {
                "total_vectors": stats.total_vector_count,
                "namespace_stats": dict(stats.namespaces) if stats.namespaces else {},
                "dimension": stats.dimension,
                "index_fullness": stats.index_fullness
            }
        except Exception as e:
            return {"error": str(e)}
    
    def delete_all(self, confirm: bool = False) -> bool:
        """Delete all vectors in the namespace (use with caution)."""
        if not confirm:
            print("❌ Must set confirm=True to delete all vectors")
            return False
        
        if not self.index:
            print("❌ Index not initialized")
            return False
        
        try:
            self.index.delete(delete_all=True, namespace=self.namespace)
            print(f"✅ Deleted all vectors in namespace '{self.namespace}'")
            return True
        except Exception as e:
            print(f"❌ Error deleting vectors: {e}")
            return False


# Global instance
pinecone_storage = PineconeStorage()


# Helper functions for easy import
def init_index(dimension: int = 1536, create_if_not_exists: bool = True) -> bool:
    """Initialize Pinecone index."""
    return pinecone_storage.init_index(dimension, create_if_not_exists)


def upsert_chunks(chunks: List[Dict[str, Any]], batch_size: int = 100, namespace: Optional[str] = None) -> bool:
    """Upsert chunks to Pinecone."""
    return pinecone_storage.upsert_chunks(chunks, batch_size, namespace)


def query_embeddings(
    query_embedding: List[float], 
    top_k: int = 10,
    filter_dict: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """Query Pinecone for similar embeddings."""
    return pinecone_storage.query_embeddings(query_embedding, top_k, filter_dict)


def get_index_stats() -> Dict[str, Any]:
    """Get Pinecone index statistics."""
    return pinecone_storage.get_index_stats()