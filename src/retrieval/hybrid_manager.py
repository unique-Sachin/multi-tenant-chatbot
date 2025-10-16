"""Hybrid Retriever Manager for multi-tenant BM25 index management."""

import logging
from typing import Dict, Optional, Any
from datetime import datetime, timedelta
from langchain_pinecone import PineconeVectorStore

from retrieval.hybrid import HybridRetriever

logger = logging.getLogger(__name__)


class HybridRetrieverManager:
    """Manages HybridRetriever instances per namespace for true multi-tenant support.
    
    This manager creates and caches HybridRetriever instances for each namespace,
    ensuring that each organization gets its own BM25 index built from its own data.
    
    Features:
    - Lazy initialization: Creates retrievers only when needed
    - LRU-style eviction: Limits memory usage by evicting old retrievers
    - Automatic rebuild: Refreshes indices after a configurable TTL
    """
    
    def __init__(
        self,
        vector_store: PineconeVectorStore,
        max_retrievers: int = 10,
        retriever_ttl_hours: int = 24
    ):
        """Initialize the manager.
        
        Args:
            vector_store: Shared Pinecone vector store
            max_retrievers: Maximum number of cached retrievers (LRU eviction)
            retriever_ttl_hours: Hours before a retriever is considered stale
        """
        self.vector_store = vector_store
        self.max_retrievers = max_retrievers
        self.retriever_ttl = timedelta(hours=retriever_ttl_hours)
        
        # Cache: namespace -> (retriever, created_at, last_used_at)
        self._retrievers: Dict[str, tuple[HybridRetriever, datetime, datetime]] = {}
        
        logger.info(f"✅ HybridRetrieverManager initialized")
        logger.info(f"   - Max cached retrievers: {max_retrievers}")
        logger.info(f"   - Retriever TTL: {retriever_ttl_hours} hours")
    
    def get_retriever(self, namespace: str) -> HybridRetriever:
        """Get or create a HybridRetriever for the given namespace.
        
        Args:
            namespace: Pinecone namespace
            
        Returns:
            HybridRetriever: Retriever instance for this namespace
        """
        current_time = datetime.now()
        
        # Check if we have a valid cached retriever
        if namespace in self._retrievers:
            retriever, created_at, last_used_at = self._retrievers[namespace]
            
            # Check if it's still fresh
            if current_time - created_at < self.retriever_ttl:
                # Update last used time
                self._retrievers[namespace] = (retriever, created_at, current_time)
                logger.info(f"♻️  Using cached HybridRetriever for namespace: '{namespace}'")
                return retriever
            else:
                # Stale - remove it
                logger.info(f"🗑️  Removing stale HybridRetriever for namespace: '{namespace}'")
                del self._retrievers[namespace]
        
        # Create new retriever
        logger.info(f"🔨 Creating new HybridRetriever for namespace: '{namespace}'")
        retriever = HybridRetriever(self.vector_store, namespace=namespace)
        
        # Add to cache
        self._retrievers[namespace] = (retriever, current_time, current_time)
        
        # Evict oldest if we exceed max_retrievers
        if len(self._retrievers) > self.max_retrievers:
            self._evict_oldest()
        
        logger.info(f"✅ HybridRetriever ready for namespace: '{namespace}'")
        logger.info(f"   - Total cached retrievers: {len(self._retrievers)}")
        
        return retriever
    
    def _evict_oldest(self) -> None:
        """Evict the least recently used retriever."""
        if not self._retrievers:
            return
        
        # Find the namespace with the oldest last_used_at
        oldest_namespace = min(
            self._retrievers.items(),
            key=lambda x: x[1][2]  # x[1][2] is last_used_at
        )[0]
        
        logger.info(f"♻️  Evicting least recently used retriever: '{oldest_namespace}'")
        del self._retrievers[oldest_namespace]
    
    def clear_namespace(self, namespace: str) -> bool:
        """Clear cached retriever for a specific namespace.
        
        Useful after re-ingestion to force rebuild of BM25 index.
        
        Args:
            namespace: Namespace to clear
            
        Returns:
            bool: True if namespace was cached and cleared
        """
        if namespace in self._retrievers:
            logger.info(f"🗑️  Clearing cached retriever for namespace: '{namespace}'")
            del self._retrievers[namespace]
            return True
        return False
    
    def clear_all(self) -> None:
        """Clear all cached retrievers."""
        count = len(self._retrievers)
        logger.info(f"🗑️  Clearing all {count} cached retrievers")
        self._retrievers.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about cached retrievers.
        
        Returns:
            Dict with stats: namespaces, count, oldest, newest
        """
        if not self._retrievers:
            return {
                "count": 0,
                "namespaces": [],
                "oldest": None,
                "newest": None
            }
        
        current_time = datetime.now()
        stats = {
            "count": len(self._retrievers),
            "namespaces": list(self._retrievers.keys()),
            "max_capacity": self.max_retrievers,
            "retrievers": {}
        }
        
        for namespace, (retriever, created_at, last_used_at) in self._retrievers.items():
            age_hours = (current_time - created_at).total_seconds() / 3600
            idle_hours = (current_time - last_used_at).total_seconds() / 3600
            
            stats["retrievers"][namespace] = {
                "age_hours": round(age_hours, 2),
                "idle_hours": round(idle_hours, 2),
                "is_stale": age_hours >= self.retriever_ttl.total_seconds() / 3600
            }
        
        return stats
