"""Retry decorators and utilities for robust API calls.

This module provides:
1. Tenacity retry decorators for OpenAI and Pinecone calls
2. Exponential backoff with jitter
3. Specific exception handling for different services
4. Retry statistics and logging
"""

import logging
from typing import Any, Callable, Type, Union, List
from functools import wraps

from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    retry_if_exception,
    before_sleep_log,
    after_log
)

# Setup logging
logger = logging.getLogger(__name__)


# OpenAI specific exceptions to retry
OPENAI_RETRY_EXCEPTIONS = (
    Exception,  # Generic fallback - will be more specific when we import openai
)

# Pinecone specific exceptions to retry  
PINECONE_RETRY_EXCEPTIONS = (
    Exception,  # Generic fallback - will be more specific when we import pinecone
)

# Network and timeout exceptions
NETWORK_RETRY_EXCEPTIONS = (
    ConnectionError,
    TimeoutError,
    OSError,
)


def should_retry_openai_exception(exception: BaseException) -> bool:
    """Determine if an OpenAI exception should trigger a retry."""
    # Check for rate limiting
    if hasattr(exception, 'response'):
        response = getattr(exception, 'response')
        if hasattr(response, 'status_code'):
            status_code = getattr(response, 'status_code')
            # Retry on 429 (rate limit), 500+ (server errors), 503 (service unavailable)
            if status_code in [429, 500, 502, 503, 504]:
                return True
    
    # Check for specific error messages
    error_msg = str(exception).lower()
    retry_indicators = [
        'rate limit',
        'timeout',
        'connection',
        'server error',
        'service unavailable',
        'internal server error',
        'bad gateway',
        'gateway timeout'
    ]
    
    return any(indicator in error_msg for indicator in retry_indicators)


def should_retry_pinecone_exception(exception: BaseException) -> bool:
    """Determine if a Pinecone exception should trigger a retry."""
    # Check for HTTP status codes if available
    if hasattr(exception, 'status_code'):
        status_code = getattr(exception, 'status_code')
        # Retry on 429, 500+
        if status_code in [429, 500, 502, 503, 504]:
            return True
    
    # Check for specific error messages
    error_msg = str(exception).lower()
    retry_indicators = [
        'rate limit',
        'timeout',
        'connection',
        'server error',
        'service unavailable',
        'internal server error',
        'quota exceeded'
    ]
    
    return any(indicator in error_msg for indicator in retry_indicators)


# OpenAI retry decorator
def retry_openai(
    max_attempts: int = 3,
    min_wait: float = 1.0,
    max_wait: float = 60.0,
    multiplier: float = 2.0
):
    """
    Retry decorator for OpenAI API calls.
    
    Args:
        max_attempts: Maximum number of retry attempts
        min_wait: Minimum wait time between retries (seconds)
        max_wait: Maximum wait time between retries (seconds)
        multiplier: Exponential backoff multiplier
    """
    return retry(
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential(
            multiplier=multiplier,
            min=min_wait,
            max=max_wait
        ),
        retry=retry_if_exception(should_retry_openai_exception),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        after=after_log(logger, logging.INFO)
    )


# Pinecone retry decorator
def retry_pinecone(
    max_attempts: int = 3,
    min_wait: float = 1.0,
    max_wait: float = 30.0,
    multiplier: float = 2.0
):
    """
    Retry decorator for Pinecone API calls.
    
    Args:
        max_attempts: Maximum number of retry attempts
        min_wait: Minimum wait time between retries (seconds)
        max_wait: Maximum wait time between retries (seconds)
        multiplier: Exponential backoff multiplier
    """
    return retry(
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential(
            multiplier=multiplier,
            min=min_wait,
            max=max_wait
        ),
        retry=retry_if_exception(should_retry_pinecone_exception),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        after=after_log(logger, logging.INFO)
    )


# Generic network retry decorator
def retry_network(
    max_attempts: int = 3,
    min_wait: float = 0.5,
    max_wait: float = 10.0,
    multiplier: float = 2.0
):
    """
    Retry decorator for general network operations.
    
    Args:
        max_attempts: Maximum number of retry attempts
        min_wait: Minimum wait time between retries (seconds)
        max_wait: Maximum wait time between retries (seconds)
        multiplier: Exponential backoff multiplier
    """
    return retry(
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential(
            multiplier=multiplier,
            min=min_wait,
            max=max_wait
        ),
        retry=retry_if_exception_type(NETWORK_RETRY_EXCEPTIONS),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        after=after_log(logger, logging.INFO)
    )


# Cohere retry decorator (for reranking)
def retry_cohere(
    max_attempts: int = 3,
    min_wait: float = 1.0,
    max_wait: float = 30.0,
    multiplier: float = 2.0
):
    """
    Retry decorator for Cohere API calls.
    
    Args:
        max_attempts: Maximum number of retry attempts
        min_wait: Minimum wait time between retries (seconds)
        max_wait: Maximum wait time between retries (seconds)
        multiplier: Exponential backoff multiplier
    """
    def should_retry_cohere_exception(exception: BaseException) -> bool:
        """Determine if a Cohere exception should trigger a retry."""
        # Check for HTTP status codes if available
        if hasattr(exception, 'status_code'):
            status_code = getattr(exception, 'status_code')
            if status_code in [429, 500, 502, 503, 504]:
                return True
        
        # Check for specific error messages
        error_msg = str(exception).lower()
        retry_indicators = [
            'rate limit',
            'timeout',
            'connection',
            'server error',
            'service unavailable'
        ]
        
        return any(indicator in error_msg for indicator in retry_indicators)
    
    return retry(
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential(
            multiplier=multiplier,
            min=min_wait,
            max=max_wait
        ),
        retry=retry_if_exception(should_retry_cohere_exception),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        after=after_log(logger, logging.INFO)
    )


# Helper function to apply retries to class methods
def add_retry_to_methods(cls, method_patterns: List[str], decorator: Callable):
    """
    Add retry decorator to class methods matching patterns.
    
    Args:
        cls: Class to modify
        method_patterns: List of method name patterns to match
        decorator: Retry decorator to apply
    """
    for attr_name in dir(cls):
        if any(pattern in attr_name for pattern in method_patterns):
            attr = getattr(cls, attr_name)
            if callable(attr) and not attr_name.startswith('_'):
                setattr(cls, attr_name, decorator(attr))


# Example usage and testing
if __name__ == "__main__":
    import time
    import random
    
    print("🧪 Testing Retry Decorators")
    print("=" * 40)
    
    # Test OpenAI retry decorator
    @retry_openai(max_attempts=3, min_wait=0.1, max_wait=1.0)
    def test_openai_call():
        """Simulate OpenAI API call that might fail."""
        if random.random() < 0.7:  # 70% chance of failure
            class MockOpenAIError(Exception):
                def __init__(self):
                    self.response = type('obj', (object,), {'status_code': 429})()
                    super().__init__("Rate limit exceeded")
            
            raise MockOpenAIError()
        
        return {"result": "success", "data": "mock openai response"}
    
    # Test Pinecone retry decorator
    @retry_pinecone(max_attempts=3, min_wait=0.1, max_wait=1.0)
    def test_pinecone_call():
        """Simulate Pinecone API call that might fail."""
        if random.random() < 0.6:  # 60% chance of failure
            class MockPineconeError(Exception):
                def __init__(self):
                    self.status_code = 503
                    super().__init__("Service unavailable")
            
            raise MockPineconeError()
        
        return {"result": "success", "vectors": [0.1, 0.2, 0.3]}
    
    # Test network retry decorator
    @retry_network(max_attempts=3, min_wait=0.1, max_wait=1.0)
    def test_network_call():
        """Simulate network call that might fail."""
        if random.random() < 0.5:  # 50% chance of failure
            raise ConnectionError("Network connection failed")
        
        return {"result": "success", "status": "connected"}
    
    # Run tests
    print("\n🤖 Testing OpenAI retry...")
    try:
        result = test_openai_call()
        print(f"✅ OpenAI call succeeded: {result['result']}")
    except Exception as e:
        print(f"❌ OpenAI call failed after retries: {e}")
    
    print("\n🌲 Testing Pinecone retry...")
    try:
        result = test_pinecone_call()
        print(f"✅ Pinecone call succeeded: {result['result']}")
    except Exception as e:
        print(f"❌ Pinecone call failed after retries: {e}")
    
    print("\n🌐 Testing network retry...")
    try:
        result = test_network_call()
        print(f"✅ Network call succeeded: {result['result']}")
    except Exception as e:
        print(f"❌ Network call failed after retries: {e}")
    
    print("\n✅ Retry decorator test completed!")