"""
Enhanced Streaming Manager with Intent Classification

Integrates custom ML model for intent-based contextual loading messages.
"""

import asyncio
import json
import time
import random
import hashlib
from typing import Dict, List, Optional, AsyncGenerator
import aiohttp
from redis import Redis


class IntentClassifierClient:
    """Client for calling intent classification microservice."""
    
    def __init__(self, service_url: str = "http://localhost:8002", timeout: float = 0.2):
        self.service_url = service_url.rstrip('/')
        self.timeout = timeout
        self.session: Optional[aiohttp.ClientSession] = None
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session."""
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession()
        return self.session
    
    async def classify(self, query: str) -> Dict[str, any]:
        """
        Classify query intent using ML service.
        
        Returns:
            {"intent": "services", "confidence": 0.95, "latency_ms": 23}
        """
        try:
            session = await self._get_session()
            
            async with session.post(
                f"{self.service_url}/classify",
                json={"query": query},
                timeout=aiohttp.ClientTimeout(total=self.timeout)
            ) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    return None
        
        except Exception as e:
            print(f"⚠️  Intent classification failed: {e}")
            return None
    
    async def close(self):
        """Close aiohttp session."""
        if self.session and not self.session.closed:
            await self.session.close()


class EnhancedStreamManager:
    """Manages enhanced streaming with smart loading states."""
    
    def __init__(
        self, 
        intent_service_url: str = "http://localhost:8002",
        redis_client: Optional[Redis] = None,
        enable_fallback: bool = True
    ):
        self.intent_client = IntentClassifierClient(intent_service_url)
        self.redis = redis_client
        self.enable_fallback = enable_fallback
        
        # Intent-based loading messages
        self.loading_templates = {
            "services": [
                "Analyzing your service inquiry...",
                "Searching our service offerings and capabilities...",
                "Reviewing relevant case studies and expertise...",
            ],
            "technology": [
                "Exploring our technology stack...",
                "Gathering technical documentation and expertise...",
                "Finding relevant technical capabilities...",
            ],
            "contact": [
                "Looking up contact information...",
                "Finding the best way to reach us...",
                "Gathering contact details...",
            ],
            "company_info": [
                "Retrieving company information...",
                "Searching company resources and details...",
                "Gathering relevant company data...",
            ],
            "meta": [
                "Understanding your question about me...",
                "Checking my capabilities...",
                "Preparing information about how I work...",
            ],
            "out_of_scope": [
                "Analyzing your question...",
                "Checking knowledge base...",
                "Searching for relevant information...",
            ],
            "general": [
                "Understanding your question...",
                "Searching knowledge base...",
                "Gathering relevant information...",
            ]
        }
        
        # Fallback keyword-based classification
        self.intent_keywords = {
            "services": ["service", "offer", "provide", "solution", "development", "project", "build"],
            "technology": ["tech", "stack", "language", "framework", "ai", "ml", "react", "node", "python"],
            "contact": ["contact", "email", "phone", "reach", "address", "url", "call", "message"],
            "company_info": ["company", "about", "office", "location", "team", "size", "founded"],
            "meta": ["you", "chatbot", "bot", "model", "capability", "ask", "who are you"],
        }
        
        self.query_cache_ttl = 600  # 10 minutes
    
    async def classify_intent(self, query: str) -> Dict[str, any]:
        """
        Classify intent with caching and fallback.
        
        Priority:
        1. Check Redis cache
        2. Call ML service
        3. Fallback to keyword matching
        """
        # Check cache first
        cache_key = f"intent:{self._hash_query(query)}"
        
        if self.redis:
            try:
                cached = self.redis.get(cache_key)
                if cached:
                    result = json.loads(cached)
                    result["source"] = "cache"
                    return result
            except Exception as e:
                print(f"⚠️  Cache read error: {e}")
        
        # Try ML service
        ml_result = await self.intent_client.classify(query)
        
        if ml_result and ml_result.get("confidence", 0) > 0.5:
            result = {
                "intent": ml_result["intent"],
                "confidence": ml_result["confidence"],
                "latency_ms": ml_result.get("latency_ms", 0),
                "source": "ml_model"
            }
            
            # Cache for future use
            if self.redis:
                try:
                    self.redis.setex(
                        cache_key,
                        self.query_cache_ttl,
                        json.dumps(result)
                    )
                except Exception as e:
                    print(f"⚠️  Cache write error: {e}")
            
            return result
        
        # Fallback to keyword matching
        if self.enable_fallback:
            intent = self._keyword_classify(query)
            result = {
                "intent": intent,
                "confidence": 0.6,
                "latency_ms": 1,
                "source": "fallback"
            }
            return result
        
        # Default
        return {
            "intent": "general",
            "confidence": 0.5,
            "latency_ms": 0,
            "source": "default"
        }
    
    def _keyword_classify(self, query: str) -> str:
        """Simple keyword-based classification as fallback."""
        query_lower = query.lower()
        
        intent_scores = {}
        for intent, keywords in self.intent_keywords.items():
            score = sum(1 for keyword in keywords if keyword in query_lower)
            if score > 0:
                intent_scores[intent] = score
        
        if intent_scores:
            return max(intent_scores, key=intent_scores.get)
        
        return "general"
    
    def _hash_query(self, query: str) -> str:
        """Generate hash for query caching."""
        return hashlib.md5(query.lower().encode()).hexdigest()[:16]
    
    def get_loading_messages(self, intent: str) -> List[str]:
        """Get contextual loading messages for an intent."""
        return self.loading_templates.get(intent, self.loading_templates["general"])
    
    async def stream_loading_messages(
        self, 
        intent: str,
        delay: float = 1.0
    ) -> AsyncGenerator[str, None]:
        """
        Stream contextual loading messages based on intent.
        
        Yields JSON-formatted messages with progressive updates.
        """
        messages = self.get_loading_messages(intent)
        
        for i, message in enumerate(messages):
            yield json.dumps({
                'type': 'loading',
                'message': message,
                'step': i + 1,
                'total_steps': len(messages),
                'intent': intent
            })
            await asyncio.sleep(delay)
    
    async def stream_retrieval_progress(
        self,
        doc_count: int,
        partition_name: str
    ) -> str:
        """Generate real-time retrieval progress message."""
        return json.dumps({
            'type': 'retrieval_progress',
            'message': f"Found {doc_count} relevant documents in knowledge base",
            'metadata': {
                'doc_count': doc_count,
                'partition': partition_name
            }
        })
    
    async def stream_reranking_progress(
        self,
        input_count: int,
        output_count: int
    ) -> str:
        """Generate reranking progress message."""
        return json.dumps({
            'type': 'reranking_progress',
            'message': f"Analyzing {input_count} documents → Selecting top {output_count}",
            'metadata': {
                'input': input_count,
                'output': output_count
            }
        })
    
    async def stream_generation_preview(self) -> str:
        """Generate message before LLM streaming begins."""
        return json.dumps({
            'type': 'generation_preview',
            'message': "✍️ Crafting your personalized answer...",
            'metadata': {
                'step': 'generation'
            }
        })
    
    async def close(self):
        """Cleanup resources."""
        await self.intent_client.close()

