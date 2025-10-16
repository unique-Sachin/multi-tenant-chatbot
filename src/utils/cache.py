"""Redis caching utilities for speed and performance optimization.

This module provides:
1. Query->retrieval result caching (TTL 10 minutes)  
2. Question+docIDs+promptHash->final answer caching (TTL 60 minutes)
3. Robust Redis connection handling with fallback
4. Cache key generation and hash utilities
"""

import os
import json
import hashlib
import pickle
import logging
from typing import Any, Optional, Dict, List, Union
from dataclasses import dataclass
from datetime import datetime, timedelta

import redis
from langchain.schema import Document

# Setup logging
logger = logging.getLogger(__name__)


@dataclass
class CacheConfig:
    """Cache configuration settings."""
    retrieval_ttl: int = 600  # 10 minutes
    answer_ttl: int = 3600    # 60 minutes  
    enabled: bool = True
    redis_url: str = "redis://localhost:6379/0"


class CacheManager:
    """Redis-based cache manager for retrieval results and final answers."""
    
    def __init__(self, config: Optional[CacheConfig] = None):
        self.config = config or CacheConfig()
        self.redis_client: Optional[redis.Redis] = None
        self.enabled = False
        
        # Initialize Redis if caching is enabled
        if self.config.enabled:
            self._init_redis()
    
    def _init_redis(self) -> None:
        """Initialize Redis client with robust error handling."""
        try:
            redis_url = os.getenv('REDIS_URL', self.config.redis_url)
            self.redis_client = redis.from_url(redis_url, decode_responses=False)
            
            # Test connection
            self.redis_client.ping()
            self.enabled = True
            logger.info("✅ Cache manager initialized with Redis")
            
        except Exception as e:
            logger.warning(f"⚠️  Redis cache unavailable: {e}")
            self.redis_client = None
            self.enabled = False
    
    def _generate_cache_key(self, prefix: str, *args) -> str:
        """Generate a consistent cache key from arguments."""
        # Create hash from all arguments
        content = json.dumps(args, sort_keys=True, default=str)
        hash_obj = hashlib.sha256(content.encode('utf-8'))
        hash_str = hash_obj.hexdigest()[:16]  # Short hash
        
        return f"{prefix}:{hash_str}"
    
    def _serialize_documents(self, docs: List[Document]) -> bytes:
        """Serialize documents for Redis storage."""
        serializable_docs = []
        for doc in docs:
            serializable_docs.append({
                'page_content': doc.page_content,
                'metadata': doc.metadata
            })
        return pickle.dumps(serializable_docs)
    
    def _deserialize_documents(self, data: bytes) -> List[Document]:
        """Deserialize documents from Redis storage."""
        serializable_docs = pickle.loads(data)
        documents = []
        for doc_data in serializable_docs:
            documents.append(Document(
                page_content=doc_data['page_content'],
                metadata=doc_data['metadata']
            ))
        return documents
    
    def get_retrieval_cache(self, query: str, retrieval_params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Get cached retrieval results for a query.
        
        Args:
            query: Search query
            retrieval_params: Parameters like k, method, etc.
            
        Returns:
            Cached retrieval result or None if not found
        """
        if not self.enabled or not self.redis_client:
            return None
        
        try:
            cache_key = self._generate_cache_key("retrieval", query, retrieval_params)
            cached_data_bytes = self.redis_client.get(cache_key)
            
            if cached_data_bytes and isinstance(cached_data_bytes, bytes):
                result = pickle.loads(cached_data_bytes)
                logger.info(f"🎯 Cache hit for retrieval: {query[:50]}...")
                return result
            
            logger.debug(f"📭 Cache miss for retrieval: {query[:50]}...")
            return None
            
        except Exception as e:
            logger.warning(f"⚠️  Cache retrieval failed: {e}")
            return None
    
    def set_retrieval_cache(
        self, 
        query: str, 
        retrieval_params: Dict[str, Any], 
        result: Dict[str, Any]
    ) -> bool:
        """
        Cache retrieval results for a query.
        
        Args:
            query: Search query
            retrieval_params: Parameters like k, method, etc.
            result: Retrieval result to cache
            
        Returns:
            True if cached successfully, False otherwise
        """
        if not self.enabled or not self.redis_client:
            return False
        
        try:
            cache_key = self._generate_cache_key("retrieval", query, retrieval_params)
            cached_data = pickle.dumps(result)
            
            success = self.redis_client.setex(
                cache_key, 
                self.config.retrieval_ttl, 
                cached_data
            )
            
            if success:
                logger.info(f"💾 Cached retrieval result: {query[:50]}...")
                return True
                
        except Exception as e:
            logger.warning(f"⚠️  Cache storage failed: {e}")
        
        return False
    
    def get_answer_cache(
        self, 
        question: str, 
        doc_ids: List[str], 
        prompt_hash: str
    ) -> Optional[str]:
        """
        Get cached final answer.
        
        Args:
            question: User question
            doc_ids: List of document IDs used for context
            prompt_hash: Hash of the prompt template
            
        Returns:
            Cached answer or None if not found
        """
        if not self.enabled or not self.redis_client:
            return None
        
        try:
            cache_key = self._generate_cache_key("answer", question, doc_ids, prompt_hash)
            cached_answer_bytes = self.redis_client.get(cache_key)
            
            if cached_answer_bytes and isinstance(cached_answer_bytes, bytes):
                answer = cached_answer_bytes.decode('utf-8')
                logger.info(f"🎯 Cache hit for answer: {question[:50]}...")
                return answer
            
            logger.debug(f"📭 Cache miss for answer: {question[:50]}...")
            return None
            
        except Exception as e:
            logger.warning(f"⚠️  Answer cache retrieval failed: {e}")
            return None
    
    def set_answer_cache(
        self, 
        question: str, 
        doc_ids: List[str], 
        prompt_hash: str, 
        answer: str
    ) -> bool:
        """
        Cache final answer.
        
        Args:
            question: User question
            doc_ids: List of document IDs used for context
            prompt_hash: Hash of the prompt template
            answer: Final answer to cache
            
        Returns:
            True if cached successfully, False otherwise
        """
        if not self.enabled or not self.redis_client:
            return False
        
        try:
            cache_key = self._generate_cache_key("answer", question, doc_ids, prompt_hash)
            
            success = self.redis_client.setex(
                cache_key, 
                self.config.answer_ttl, 
                answer.encode('utf-8')
            )
            
            if success:
                logger.info(f"💾 Cached answer: {question[:50]}...")
                return True
                
        except Exception as e:
            logger.warning(f"⚠️  Answer cache storage failed: {e}")
        
        return False
    
    def generate_prompt_hash(self, prompt_template: str, **kwargs) -> str:
        """Generate hash for prompt template and variables."""
        prompt_content = prompt_template + json.dumps(kwargs, sort_keys=True)
        return hashlib.sha256(prompt_content.encode('utf-8')).hexdigest()[:16]
    
    def clear_cache(self, pattern: Optional[str] = None) -> int:
        """
        Clear cache entries.
        
        Args:
            pattern: Redis key pattern to match (e.g., "retrieval:*")
            
        Returns:
            Number of keys deleted
        """
        if not self.enabled:
            return 0
        
        try:
            if pattern:
                keys = self.redis_client.keys(pattern)
            else:
                keys = self.redis_client.keys("retrieval:*") + self.redis_client.keys("answer:*")
            
            if keys:
                deleted = self.redis_client.delete(*keys)
                logger.info(f"🗑️  Cleared {deleted} cache entries")
                return deleted
            
            return 0
            
        except Exception as e:
            logger.warning(f"⚠️  Cache clear failed: {e}")
            return 0
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        if not self.enabled:
            return {"enabled": False, "error": "Redis not available"}
        
        try:
            info = self.redis_client.info()
            return {
                "enabled": True,
                "connected_clients": info.get("connected_clients", 0),
                "used_memory_human": info.get("used_memory_human", "unknown"),
                "keyspace_hits": info.get("keyspace_hits", 0),
                "keyspace_misses": info.get("keyspace_misses", 0),
                "hit_rate": (
                    info.get("keyspace_hits", 0) / 
                    max(info.get("keyspace_hits", 0) + info.get("keyspace_misses", 0), 1)
                ) * 100
            }
            
        except Exception as e:
            return {"enabled": False, "error": str(e)}


# Cache utilities
def create_cache_manager() -> CacheManager:
    """Create cache manager based on environment configuration."""
    cache_enabled = os.getenv('CACHE_ENABLED', 'true').lower() == 'true'
    redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379/0')
    
    config = CacheConfig(
        enabled=cache_enabled,
        redis_url=redis_url
    )
    
    return CacheManager(config)


# Example usage and testing
if __name__ == "__main__":
    import time
    from langchain.schema import Document
    
    print("🧪 Testing Cache Manager")
    print("=" * 40)
    
    # Initialize cache manager
    cache = create_cache_manager()
    
    if not cache.enabled:
        print("❌ Cache not enabled or Redis not available")
        exit(1)
    
    # Test retrieval caching
    print("\n📥 Testing retrieval cache...")
    
    query = "What programming languages does Zibtek use?"
    params = {"k": 20, "method": "hybrid"}
    
    # Simulate retrieval result
    test_docs = [
        Document(page_content="Zibtek uses Python and PHP", metadata={"id": "doc1"}),
        Document(page_content="Laravel and Django are our frameworks", metadata={"id": "doc2"})
    ]
    
    retrieval_result = {
        "documents": cache._serialize_documents(test_docs),
        "scores": {"doc1": 0.95, "doc2": 0.87},
        "method": "hybrid"
    }
    
    # Test cache miss
    cached = cache.get_retrieval_cache(query, params)
    print(f"Cache miss test: {cached is None}")
    
    # Test cache set
    success = cache.set_retrieval_cache(query, params, retrieval_result)
    print(f"Cache set: {success}")
    
    # Test cache hit
    cached = cache.get_retrieval_cache(query, params)
    print(f"Cache hit test: {cached is not None}")
    
    # Test answer caching
    print("\n💬 Testing answer cache...")
    
    question = "What services does Zibtek offer?"
    doc_ids = ["doc1", "doc2", "doc3"]
    prompt_hash = cache.generate_prompt_hash("Answer: {context}", context="test")
    answer = "Zibtek offers web development, mobile apps, and custom software solutions."
    
    # Test answer cache miss
    cached_answer = cache.get_answer_cache(question, doc_ids, prompt_hash)
    print(f"Answer cache miss: {cached_answer is None}")
    
    # Test answer cache set
    success = cache.set_answer_cache(question, doc_ids, prompt_hash, answer)
    print(f"Answer cache set: {success}")
    
    # Test answer cache hit
    cached_answer = cache.get_answer_cache(question, doc_ids, prompt_hash)
    print(f"Answer cache hit: {cached_answer is not None}")
    print(f"Cached answer: {cached_answer[:50]}...")
    
    # Test cache stats
    print("\n📊 Cache statistics:")
    stats = cache.get_cache_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Test cache clear
    print(f"\n🗑️  Clearing cache...")
    cleared = cache.clear_cache()
    print(f"Cleared {cleared} entries")
    
    print("\n✅ Cache manager test completed!")