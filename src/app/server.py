"""FastAPI server for Zibtek chatbot with retrieval and guardrails."""

import time
import uuid
import json
import asyncio
import hashlib
from datetime import datetime
from typing import List, Dict, Any, Optional, AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, File, UploadFile, Form
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field
import tiktoken
from dotenv import load_dotenv
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# LangChain imports
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

# Local imports
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from guards.guards import ZibtekGuards
from retrieval.retriever import ZibtekRetriever
from retrieval.rerank import rerank_with_scores
from retrieval.hybrid import HybridRetriever
from retrieval.hybrid_manager import HybridRetrieverManager
from storage.db import create_chat_log, ChatLogCreate
from storage.auth import (
    UserSignup, UserLogin, UserResponse, AuthResponse,
    create_user, authenticate_user, get_user_by_id,
    create_access_token, verify_access_token
)
from storage.conversations import (
    get_user_conversations, get_conversation_messages,
    create_or_update_session, delete_conversation
)
from storage.documents import (
    get_documents_by_org, get_documents_by_namespace,
    delete_document_record, get_document_stats
)
from utils.cache import create_cache_manager, CacheManager
from utils.retries import retry_openai, retry_pinecone, retry_cohere

import time
import uuid
import json
import asyncio
from datetime import datetime
from typing import List, Dict, Any, Optional, AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import tiktoken
from dotenv import load_dotenv

# LangChain imports
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

# Local imports
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from guards.guards import ZibtekGuards
from retrieval.retriever import ZibtekRetriever
from retrieval.rerank import rerank_with_scores
from retrieval.hybrid import HybridRetriever
from storage.db import create_chat_log, ChatLogCreate

load_dotenv()

# Configuration
RERANK_ENABLED = os.getenv("RERANK_ENABLED", "true").lower() == "true"
CACHE_ENABLED = os.getenv("CACHE_ENABLED", "true").lower() == "true"
RATELIMIT_ENABLED = os.getenv("RATELIMIT_ENABLED", "true").lower() == "true"

# Rate limiter setup
limiter = Limiter(key_func=get_remote_address)
cache_manager: Optional[CacheManager] = None


# Pydantic models
class ChatRequest(BaseModel):
    """Chat request model."""
    question: str = Field(..., min_length=1, max_length=1000, description="User's question")
    session_id: Optional[str] = Field(None, description="Optional session identifier")
    namespace: Optional[str] = Field("zibtek", description="Organization namespace for retrieval")


class ChatStreamRequest(BaseModel):
    """Chat streaming request model."""
    question: str = Field(..., min_length=1, max_length=1000, description="User's question")
    session_id: Optional[str] = Field(None, description="Optional session identifier")
    namespace: Optional[str] = Field("zibtek", description="Organization namespace for retrieval")


class Citation(BaseModel):
    """Citation model."""
    url: str
    title: Optional[str] = None


class ChatResponse(BaseModel):
    """Chat response model."""
    answer: str
    citations: List[str]
    session_id: str
    processing_time_ms: int
    is_out_of_scope: bool = False
    rerank_scores: Dict[str, float] = {}
    hybrid_scores: Dict[str, float] = {}
    retrieval_method: str = "vector_only"
    retrieval_steps: Dict[str, Any] = {}  # Detailed step-by-step retrieval information


# Global instances
guards: Optional[ZibtekGuards] = None
retriever: Optional[ZibtekRetriever] = None
hybrid_retriever: Optional[HybridRetriever] = None  # Deprecated: kept for backward compatibility
hybrid_manager: Optional[HybridRetrieverManager] = None  # New: multi-tenant hybrid manager
llm: Optional[ChatOpenAI] = None
tokenizer: Optional[tiktoken.Encoding] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize services on startup."""
    global guards, retriever, hybrid_retriever, hybrid_manager, llm, tokenizer, cache_manager
    
    print("🚀 Starting Zibtek Chatbot Server...")
    
    try:
        # Initialize cache manager
        if CACHE_ENABLED:
            print("💾 Initializing cache manager...")
            cache_manager = create_cache_manager()
        
        # Initialize guardrails
        print("🛡️  Initializing guardrails...")
        guards = ZibtekGuards()
        
        # Initialize retriever
        print("🔍 Initializing retriever...")
        retriever = ZibtekRetriever()
        
        # Initialize hybrid retriever manager if enabled (multi-tenant support)
        hybrid_enabled = os.getenv('HYBRID_ENABLED', 'false').lower() == 'true'
        if hybrid_enabled:
            print("🔀 Initializing hybrid retriever manager (multi-tenant)...")
            hybrid_manager = HybridRetrieverManager(
                vector_store=retriever.vector_store,
                max_retrievers=10,  # Cache up to 10 namespace-specific retrievers
                retriever_ttl_hours=24  # Rebuild indices after 24 hours
            )
            print("✅ Hybrid manager ready - will create retrievers on-demand per namespace")
        
        # Initialize LLM
        print("🤖 Initializing LLM...")
        llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0,
            timeout=30
        )
        
        # Initialize tokenizer for cost estimation
        tokenizer = tiktoken.encoding_for_model("gpt-4o-mini")
        
        print("✅ All services initialized successfully!")
        
    except Exception as e:
        print(f"❌ Failed to initialize services: {e}")
        raise
    
    yield
    
    print("👋 Shutting down Zibtek Chatbot Server...")


# Create FastAPI app
app = FastAPI(
    title="Zibtek Chatbot API",
    description="AI-powered chatbot for Zibtek software development services",
    version="1.0.0",
    lifespan=lifespan
)

# Add rate limiting if enabled
if RATELIMIT_ENABLED:
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


def count_tokens(text: str) -> int:
    """Count tokens in text."""
    if not tokenizer:
        return len(text) // 4  # Rough estimate
    return len(tokenizer.encode(text))


def estimate_cost(input_tokens: int, output_tokens: int) -> float:
    """Estimate API cost based on token counts.
    
    GPT-4o-mini pricing (as of 2024):
    - Input: $0.00015 per 1K tokens
    - Output: $0.0006 per 1K tokens
    """
    input_cost = (input_tokens / 1000) * 0.00015
    output_cost = (output_tokens / 1000) * 0.0006
    return input_cost + output_cost


def build_context_from_docs(docs: List[Any]) -> str:
    """Build context string from retrieved documents."""
    context_blocks = []
    
    for i, doc in enumerate(docs, 1):
        url = doc.metadata.get('url', 'Unknown URL')
        title = doc.metadata.get('title', 'Untitled')
        
        # Sanitize and truncate content
        content = guards.sanitize(doc.page_content) if guards else doc.page_content
        content = content.strip()
        
        # Limit content length to prevent token overflow
        if len(content) > 800:
            content = content[:800] + "..."
        
        context_block = f"""CONTEXT {i}:
URL: {url}
Title: {title}
Content: {content}

"""
        context_blocks.append(context_block)
    
    return "\n".join(context_blocks)


def extract_citations(docs: List[Any]) -> List[str]:
    """Extract unique URLs from documents for citations."""
    urls = []
    seen = set()
    
    for doc in docs:
        url = doc.metadata.get('url')
        if url and url not in seen:
            urls.append(url)
            seen.add(url)
    
    return urls


async def log_chat_interaction(
    request: ChatRequest,
    response: ChatResponse,
    input_tokens: int,
    output_tokens: int,
    cost: float,
    user_ip: str,
    user_id: str,
    namespace: str,
    rerank_scores: Optional[Dict[str, float]] = None
) -> None:
    """Log chat interaction to database."""
    try:
        # Create conversation session
        create_or_update_session(user_id, response.session_id, namespace)
        
        chat_log = ChatLogCreate(
            session_id=response.session_id,
            user_id=user_id,
            namespace=namespace,
            user_query=request.question,
            answer=response.answer,
            is_oos=response.is_out_of_scope,
            latency_ms=response.processing_time_ms,
            cost_cents=int(cost * 100),  # Convert to cents
            citations=[{"url": url} for url in response.citations],
            model="gpt-4o-mini",
            retrieved_urls=response.citations,
            rerank_scores=rerank_scores or {},
            retrieval_steps=response.retrieval_steps
        )
        
        create_chat_log(chat_log)
        print(f"✅ Logged chat interaction for user {user_id}: {response.session_id}")
        
    except Exception as e:
        print(f"❌ Failed to log chat interaction: {e}")
        # Don't fail the request if logging fails


@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "service": "Zibtek Chatbot API",
        "status": "healthy",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat()
    }


@app.get("/health")
async def health():
    """Detailed health check."""
    health_status = {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "services": {}
    }
    
    # Check guardrails
    try:
        if guards:
            guards.sanitize("test")
            health_status["services"]["guardrails"] = "healthy"
        else:
            health_status["services"]["guardrails"] = "not_initialized"
    except Exception as e:
        health_status["services"]["guardrails"] = f"error: {str(e)}"
    
    # Check retriever
    try:
        if retriever:
            health_status["services"]["retriever"] = "healthy"
        else:
            health_status["services"]["retriever"] = "not_initialized"
    except Exception as e:
        health_status["services"]["retriever"] = f"error: {str(e)}"
    
    # Check LLM
    try:
        if llm:
            health_status["services"]["llm"] = "healthy"
        else:
            health_status["services"]["llm"] = "not_initialized"
    except Exception as e:
        health_status["services"]["llm"] = f"error: {str(e)}"
    
    # Overall status
    if any("error" in status for status in health_status["services"].values()):
        health_status["status"] = "degraded"
    elif any("not_initialized" in status for status in health_status["services"].values()):
        health_status["status"] = "initializing"
    
    return health_status


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, http_request: Request):
    """Main chat endpoint with caching and rate limiting. Requires authentication."""
    start_time = time.time()
    
    # Authenticate user
    user = await get_current_user(http_request)
    
    # Generate session ID if not provided
    session_id = request.session_id or str(uuid.uuid4())
    
    # Use namespace from request or default to 'zibtek'
    namespace = request.namespace or "zibtek"
    
    # Get user IP for logging
    user_ip = http_request.client.host if http_request.client else "unknown"
    
    # Initialize response
    response = ChatResponse(
        answer="",
        citations=[],
        session_id=session_id,
        processing_time_ms=0,
        is_out_of_scope=False
    )
    
    input_tokens = 0
    output_tokens = 0
    cost = 0.0
    
    try:
        # Validate services are initialized
        if not guards or not retriever or not llm:
            raise HTTPException(
                status_code=503,
                detail="Services not properly initialized"
            )
        
        print(f"📝 Processing question: {request.question[:50]}...")
        print(f"🏢 Namespace: {namespace}")
        print(f"👤 User: {user.id}")
        
        # Step 1: Validate and sanitize question
        validation_result = guards.validate_question(request.question)
        
        if validation_result['is_blocked']:
            print(f"🛡️  Question blocked: {validation_result['block_reason']}")
            response.answer = guards.get_out_of_scope_message()
            response.is_out_of_scope = True
            response.processing_time_ms = int((time.time() - start_time) * 1000)
            
            # Log the blocked interaction
            await log_chat_interaction(
                request, response, input_tokens, output_tokens, cost, user_ip,
                user.id, namespace, rerank_scores={}
            )
            
            return response
        
        # Use sanitized question
        clean_question = validation_result['sanitized_question']
        
        # Step 1.5: Check answer cache first (if enabled and not streaming)
        cached_answer = None
        if CACHE_ENABLED and cache_manager and not getattr(request, 'stream', False):
            # Generate cache key based on question (will add doc context later)
            print("🔍 Checking answer cache...")
            # For now, we'll cache after we have doc_ids and prompt_hash
        
        # Use sanitized question
        clean_question = validation_result['sanitized_question']
        
        # Step 2: Check retrieval cache and retrieve relevant documents
        hybrid_enabled = os.getenv('HYBRID_ENABLED', 'false').lower() == 'true'
        retrieval_params = {
            "k": 20,
            "method": "hybrid" if hybrid_enabled else "vector",
            "question": clean_question
        }
        
        # Check retrieval cache first
        cached_retrieval = None
        if CACHE_ENABLED and cache_manager:
            print("🔍 Checking retrieval cache...")
            cached_retrieval = cache_manager.get_retrieval_cache(clean_question, retrieval_params)
        
        # Initialize retrieval steps tracking
        retrieval_steps = {
            "vector_search": {},
            "bm25_search": {},
            "rrf_fusion": {},
            "reranking": {},
            "method": ""
        }
        
        if cached_retrieval:
            print("🎯 Using cached retrieval results!")
            docs = cached_retrieval.get('documents', [])
            hybrid_scores = cached_retrieval.get('hybrid_scores', {})
            retrieval_method = cached_retrieval.get('method', 'cached')
            retrieval_steps = cached_retrieval.get('retrieval_steps', retrieval_steps)
            retrieval_steps["method"] = retrieval_method + " (cached)"
        elif hybrid_enabled and hybrid_manager:
            print(f"🔀 Using hybrid retrieval (vector + BM25) for namespace: '{namespace}'...")
            # Get namespace-specific hybrid retriever
            namespace_retriever = hybrid_manager.get_retriever(namespace)
            hybrid_result = namespace_retriever.hybrid_search(clean_question, k=20, namespace=namespace)
            docs = hybrid_result.documents
            hybrid_scores = hybrid_result.fusion_scores
            retrieval_method = hybrid_result.method
            
            # Capture detailed step information
            retrieval_steps["vector_search"] = {
                "count": len(hybrid_result.vector_scores),
                "scores": dict(list(hybrid_result.vector_scores.items())[:5])  # Top 5 for display
            }
            retrieval_steps["bm25_search"] = {
                "count": len(hybrid_result.bm25_scores),
                "scores": dict(list(hybrid_result.bm25_scores.items())[:5])  # Top 5 for display
            }
            retrieval_steps["rrf_fusion"] = {
                "count": len(hybrid_result.fusion_scores),
                "scores": dict(list(hybrid_result.fusion_scores.items())[:5])  # Top 5 for display
            }
            retrieval_steps["method"] = retrieval_method
            
            print(f"✅ Hybrid retrieval complete: {len(docs)} documents from namespace '{namespace}' via {retrieval_method}")
            
            # Cache retrieval results with steps
            if CACHE_ENABLED and cache_manager:
                retrieval_cache_data = {
                    'documents': docs,
                    'hybrid_scores': hybrid_scores,
                    'method': retrieval_method,
                    'retrieval_steps': retrieval_steps
                }
                cache_manager.set_retrieval_cache(clean_question, retrieval_params, retrieval_cache_data)
        else:
            print(f"🔍 Using vector-only retrieval")
            print(f"   Namespace: '{namespace}'")
            docs = retriever.retrieve(clean_question, k=20, namespace=namespace)
            hybrid_scores = {}
            retrieval_method = "vector_only"
            
            # Capture vector-only step information
            retrieval_steps["vector_search"] = {
                "count": len(docs),
                "scores": {}
            }
            retrieval_steps["method"] = retrieval_method
            
            print(f"   Retrieved {len(docs)} documents from namespace '{namespace}'")
            
            # Cache retrieval results with steps
            if CACHE_ENABLED and cache_manager:
                retrieval_cache_data = {
                    'documents': docs,
                    'hybrid_scores': hybrid_scores,
                    'method': retrieval_method,
                    'retrieval_steps': retrieval_steps
                }
                cache_manager.set_retrieval_cache(clean_question, retrieval_params, retrieval_cache_data)
        
        if not docs:
            print("❌ No relevant documents found")
            response.answer = guards.get_out_of_scope_message()
            response.is_out_of_scope = True
            response.processing_time_ms = int((time.time() - start_time) * 1000)
            
            await log_chat_interaction(
                request, response, input_tokens, output_tokens, cost, user_ip,
                user.id, namespace, rerank_scores={}
            )
            
            return response
        
        # Step 3: Rerank documents for better precision (optional)
        rerank_scores = {}
        if RERANK_ENABLED and len(docs) > 4:
            try:
                print(f"🎯 Reranking {len(docs)} documents with Cohere...")
                top_docs, rerank_scores = rerank_with_scores(clean_question, docs, top_n=4)
                print(f"✅ Reranked to top {len(top_docs)} documents")
                
                # Capture reranking step information
                retrieval_steps["reranking"] = {
                    "enabled": True,
                    "input_count": len(docs),
                    "output_count": len(top_docs),
                    "scores": dict(list(rerank_scores.items())[:5])  # Top 5 for display
                }
            except Exception as e:
                print(f"⚠️  Reranking failed, using original order: {e}")
                top_docs = docs[:4]
                rerank_scores = {}
                retrieval_steps["reranking"] = {
                    "enabled": True,
                    "error": str(e),
                    "input_count": len(docs),
                    "output_count": len(top_docs)
                }
        else:
            top_docs = docs[:4]
            if not RERANK_ENABLED:
                print("🔄 Reranking disabled via RERANK_ENABLED=false")
                retrieval_steps["reranking"] = {"enabled": False, "reason": "disabled"}
            else:
                print(f"📄 Only {len(docs)} documents, skipping rerank")
                retrieval_steps["reranking"] = {
                    "enabled": False,
                    "reason": f"insufficient_docs",
                    "input_count": len(docs)
                }
        
        print(f"📄 Using top {len(top_docs)} documents for context")
        
        # Step 4: Build context and prompt
        context = build_context_from_docs(top_docs)
        system_prompt = guards.system_prompt()
        
        # Create messages
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"""Question: {clean_question}

{context}

Please answer the question using ONLY the information provided in the CONTEXT above. Always cite the URL(s) where you found the information.""")
        ]
        
        # Count input tokens
        full_prompt = system_prompt + clean_question + context
        input_tokens = count_tokens(full_prompt)
        
        # Step 5: Generate response with LLM
        print("🤖 Generating response with LLM...")
        llm_response = llm.invoke(messages)
        answer = str(llm_response.content) if llm_response.content else "I couldn't generate a response."
        
        # Count output tokens
        output_tokens = count_tokens(answer)
        
        # Estimate cost
        cost = estimate_cost(input_tokens, output_tokens)
        
        # Step 6: Extract citations
        citations = extract_citations(top_docs)
        
        # Prepare final response
        response.answer = answer
        response.citations = citations
        response.rerank_scores = rerank_scores
        response.hybrid_scores = hybrid_scores
        response.retrieval_method = retrieval_method
        response.retrieval_steps = retrieval_steps
        response.processing_time_ms = int((time.time() - start_time) * 1000)
        
        print(f"✅ Generated response in {response.processing_time_ms}ms")
        print(f"💰 Estimated cost: ${cost:.6f}")
        
        # Step 7: Log interaction
        await log_chat_interaction(
            request, response, input_tokens, output_tokens, cost, user_ip,
            user.id, namespace, rerank_scores=rerank_scores
        )
        
        return response
        
    except Exception as e:
        print(f"❌ Error processing chat request: {e}")
        
        # Calculate processing time even for errors
        response.processing_time_ms = int((time.time() - start_time) * 1000)
        
        # Return error response
        response.answer = "I apologize, but I'm experiencing technical difficulties. Please try again later."
        response.is_out_of_scope = True
        
        # Try to log error interaction
        try:
            await log_chat_interaction(
                request, response, input_tokens, output_tokens, cost, user_ip,
                user.id, namespace, rerank_scores={}
            )
        except:
            pass  # Don't fail on logging errors
        
        raise HTTPException(
            status_code=500,
            detail="Internal server error while processing your question"
        )


@app.post("/chat/stream")
async def chat_stream(request: ChatStreamRequest, http_request: Request) -> StreamingResponse:
    """Streaming chat endpoint that yields response tokens as they're generated."""
    
    async def generate_stream():
        start_time = time.time()
        session_id = request.session_id or str(uuid.uuid4())
        
        # Authenticate user
        try:
            user = await get_current_user(http_request)
        except HTTPException:
            yield f"data: {json.dumps({'type': 'error', 'content': 'Authentication required'})}\n\n"
            return
        
        # Get user IP for logging
        user_ip = http_request.client.host if http_request.client else "unknown"
        
        try:
            # Initial metadata
            yield f"data: {json.dumps({'type': 'start', 'session_id': session_id})}\n\n"
            
            # Guardrails check
            yield f"data: {json.dumps({'type': 'status', 'message': 'Checking content safety...'})}\n\n"
            
            # Get namespace from request
            namespace = request.namespace or "zibtek"
            
            # Pass namespace to guardrails for multi-tenant scope checking
            is_oos, reason = guards.is_out_of_scope(request.question, namespace)
            if is_oos:
                warning_response = guards.get_out_of_scope_message()
                
                # Log the out-of-scope interaction with full details
                processing_time = int((time.time() - start_time) * 1000)
                
                # Create conversation session
                create_or_update_session(user.id, session_id, namespace)
                
                # Create comprehensive chat log entry for out-of-scope
                chat_log = ChatLogCreate(
                    session_id=session_id,
                    user_id=user.id,
                    namespace=namespace,
                    user_query=request.question,
                    answer=warning_response,
                    is_oos=True,
                    latency_ms=processing_time,
                    cost_cents=0,
                    citations=[],
                    model="gpt-4o-mini",
                    retrieved_urls=[],
                    rerank_scores={},
                    retrieval_steps={}
                )
                
                try:
                    create_chat_log(chat_log)
                    print(f"✅ Logged out-of-scope streaming interaction for user {user.id}: {session_id}")
                except Exception as e:
                    print(f"❌ Failed to log out-of-scope streaming interaction: {e}")
                
                # Stream the warning response
                yield f"data: {json.dumps({'type': 'status', 'message': 'Content filtered'})}\n\n"
                
                words = warning_response.split()
                for i, word in enumerate(words):
                    partial_text = " ".join(words[:i+1])
                    yield f"data: {json.dumps({'type': 'token', 'content': partial_text})}\n\n"
                    await asyncio.sleep(0.05)  # Simulate typing speed
                
                yield f"data: {json.dumps({'type': 'complete', 'citations': [], 'is_out_of_scope': True, 'processing_time_ms': int((time.time() - start_time) * 1000)})}\n\n"
                return
            
            # Retrieval
            yield f"data: {json.dumps({'type': 'status', 'message': 'Searching knowledge base...'})}\n\n"
            
            # Initialize retrieval steps tracking
            retrieval_steps = {
                "vector_search": {},
                "bm25_search": {},
                "rrf_fusion": {},
                "reranking": {},
                "method": ""
            }
            
            # Check if hybrid retrieval is enabled
            hybrid_enabled = os.getenv('HYBRID_ENABLED', 'false').lower() == 'true'
            
            if hybrid_enabled and hybrid_manager:
                print(f"🔀 Using hybrid retrieval (vector + BM25) for namespace: '{namespace}'...")
                # Get namespace-specific hybrid retriever
                namespace_retriever = hybrid_manager.get_retriever(namespace)
                hybrid_result = namespace_retriever.hybrid_search(request.question, k=20, namespace=namespace)
                docs = hybrid_result.documents
                hybrid_scores = hybrid_result.fusion_scores
                retrieval_method = hybrid_result.method
                
                # Capture detailed step information
                retrieval_steps["vector_search"] = {
                    "count": len(hybrid_result.vector_scores),
                    "scores": dict(list(hybrid_result.vector_scores.items())[:5])  # Top 5 for display
                }
                retrieval_steps["bm25_search"] = {
                    "count": len(hybrid_result.bm25_scores),
                    "scores": dict(list(hybrid_result.bm25_scores.items())[:5])  # Top 5 for display
                }
                retrieval_steps["rrf_fusion"] = {
                    "count": len(hybrid_result.fusion_scores),
                    "scores": dict(list(hybrid_result.fusion_scores.items())[:5])  # Top 5 for display
                }
                retrieval_steps["method"] = retrieval_method
                
                print(f"✅ Hybrid retrieval complete: {len(docs)} documents from namespace '{namespace}' via {retrieval_method}")
            else:
                print(f"🔍 Using vector-only retrieval for namespace: '{namespace}'")
                docs = retriever.retrieve_with_scores(request.question, k=20, namespace=namespace)
                
                # Capture vector-only step information
                retrieval_steps["vector_search"] = {
                    "count": len(docs),
                    "scores": {str(i): score for i, (doc, score) in enumerate(docs[:5])}  # Top 5 for display
                }
                retrieval_steps["method"] = "vector_only"
                
                # Convert from (doc, score) tuples to just docs for consistency
                docs = [doc for doc, score in docs]
            
            # Filter high-confidence docs and limit to top 4
            filtered_docs = docs[:4]  # Take top 4 instead of filtering by score
            
            # Step 3: Rerank documents for better precision (optional)
            rerank_scores = {}
            if RERANK_ENABLED and len(docs) > 4:
                try:
                    print(f"🎯 Reranking {len(docs)} documents with Cohere...")
                    filtered_docs, rerank_scores = rerank_with_scores(request.question, docs, top_n=4)
                    print(f"✅ Reranked to top {len(filtered_docs)} documents")
                    
                    # Capture reranking step information
                    retrieval_steps["reranking"] = {
                        "enabled": True,
                        "input_count": len(docs),
                        "output_count": len(filtered_docs),
                        "scores": dict(list(rerank_scores.items())[:5])  # Top 5 for display
                    }
                except Exception as e:
                    print(f"⚠️  Reranking failed, using original order: {e}")
                    filtered_docs = docs[:4]
                    rerank_scores = {}
                    retrieval_steps["reranking"] = {
                        "enabled": True,
                        "error": str(e),
                        "input_count": len(docs),
                        "output_count": len(filtered_docs)
                    }
            else:
                filtered_docs = docs[:4]
                if not RERANK_ENABLED:
                    print("🔄 Reranking disabled via RERANK_ENABLED=false")
                    retrieval_steps["reranking"] = {"enabled": False, "reason": "disabled"}
                else:
                    print(f"📄 Only {len(docs)} documents, skipping rerank")
                    retrieval_steps["reranking"] = {
                        "enabled": False,
                        "reason": f"insufficient_docs",
                        "input_count": len(docs)
                    }
            
            if not filtered_docs:
                no_context_response = "I don't have specific information about that topic in my knowledge base. Could you please ask about Zibtek's software development services, technologies, or company information?"
                
                words = no_context_response.split()
                for i, word in enumerate(words):
                    partial_text = " ".join(words[:i+1])
                    yield f"data: {json.dumps({'type': 'token', 'content': partial_text})}\n\n"
                    await asyncio.sleep(0.05)
                
                yield f"data: {json.dumps({'type': 'complete', 'citations': [], 'is_out_of_scope': True, 'processing_time_ms': int((time.time() - start_time) * 1000)})}\n\n"
                return
            
            # Generate response
            yield f"data: {json.dumps({'type': 'status', 'message': 'Generating response...'})}\n\n"
            
            # Prepare context and messages
            context = "\n\n".join([doc.page_content for doc in filtered_docs])
            citations = extract_citations(filtered_docs)  # Use the proper citation extraction function
            
            system_prompt = guards.system_prompt()
            user_prompt = f"Context:\n{context}\n\nQuestion: {request.question}"
            
            # Get LLM response with streaming
            messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
            
            # Stream the response
            full_response = ""
            async for chunk in llm.astream(messages):
                if hasattr(chunk, 'content') and chunk.content:
                    token = str(chunk.content)
                    full_response += token
                    yield f"data: {json.dumps({'type': 'token', 'content': full_response})}\n\n"
            
            # Sanitize and finalize
            sanitized_response = guards.sanitize(full_response)
            
            # Log the interaction with full details
            processing_time = int((time.time() - start_time) * 1000)
            
            # Create conversation session
            create_or_update_session(user.id, session_id, namespace)
            
            # Create comprehensive chat log entry
            chat_log = ChatLogCreate(
                session_id=session_id,
                user_id=user.id,
                namespace=namespace,
                user_query=request.question,
                answer=sanitized_response,
                is_oos=False,
                latency_ms=processing_time,
                cost_cents=0,  # TODO: Calculate actual cost if needed
                citations=[{"url": url} for url in citations],
                model="gpt-4o-mini",
                retrieved_urls=citations,
                rerank_scores=rerank_scores,
                retrieval_steps=retrieval_steps
            )
            
            try:
                create_chat_log(chat_log)
                print(f"✅ Logged streaming chat interaction for user {user.id}: {session_id}")
            except Exception as e:
                print(f"❌ Failed to log streaming chat interaction: {e}")
            
            # Final completion message
            yield f"data: {json.dumps({'type': 'complete', 'citations': citations, 'is_out_of_scope': False, 'processing_time_ms': processing_time, 'retrieval_steps': retrieval_steps})}\n\n"
            
        except Exception as e:
            error_msg = f"I apologize, but I encountered an error while processing your request: {str(e)}"
            yield f"data: {json.dumps({'type': 'error', 'content': error_msg})}\n\n"
    
    return StreamingResponse(
        generate_stream(),
        media_type="text/plain",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*"
        }
    )


@app.get("/stats")
async def get_stats():
    """Get chatbot usage statistics."""
    try:
        from storage.db import get_chat_log_stats
        stats = get_chat_log_stats()
        return stats
    except Exception as e:
        print(f"❌ Error getting stats: {e}")
        raise HTTPException(
            status_code=500,
            detail="Error retrieving statistics"
        )


# Error handlers
@app.exception_handler(404)
async def not_found_handler(request: Request, exc):
    return {
        "error": "Endpoint not found",
        "detail": f"The endpoint {request.url.path} does not exist",
        "available_endpoints": ["/", "/health", "/chat", "/stats"]
    }


# ===== ORGANIZATION MANAGEMENT ENDPOINTS =====

from storage.organizations import (
    OrganizationCreate, OrganizationResponse,
    WebsiteCreate, WebsiteResponse,
    IngestionJobCreate, IngestionJobResponse,
    create_organization, get_organization, list_organizations,
    create_website, get_website, list_websites, update_website_status,
    create_ingestion_job, get_ingestion_job, list_ingestion_jobs
)
from app.jobs import job_manager
from app.ingestion_runner import run_ingestion_job


@app.post("/organizations", response_model=OrganizationResponse)
async def create_org(org_data: OrganizationCreate):
    """Create a new organization."""
    org = create_organization(org_data)
    if not org:
        raise HTTPException(status_code=500, detail="Failed to create organization")
    return org


@app.get("/organizations", response_model=List[OrganizationResponse])
async def list_orgs():
    """List all organizations."""
    return list_organizations()


@app.get("/organizations/{org_id}", response_model=OrganizationResponse)
async def get_org(org_id: str):
    """Get organization by ID."""
    org = get_organization(org_id)
    if not org:
        raise HTTPException(status_code=404, detail="Organization not found")
    return org


@app.post("/organizations/{org_id}/websites", response_model=WebsiteResponse)
async def create_org_website(org_id: str, website_data: WebsiteCreate):
    """Create a new website for an organization."""
    # Verify org exists
    org = get_organization(org_id)
    if not org:
        raise HTTPException(status_code=404, detail="Organization not found")
    
    # Ensure org_id matches
    website_data.org_id = org_id
    
    website = create_website(website_data)
    if not website:
        raise HTTPException(status_code=500, detail="Failed to create website")
    return website


@app.get("/websites", response_model=List[WebsiteResponse])
async def list_all_websites(org_id: Optional[str] = None):
    """List all websites, optionally filtered by organization."""
    return list_websites(org_id)


@app.get("/websites/{website_id}", response_model=WebsiteResponse)
async def get_website_detail(website_id: str):
    """Get website details."""
    website = get_website(website_id)
    if not website:
        raise HTTPException(status_code=404, detail="Website not found")
    return website


@app.post("/websites/{website_id}/ingest")
async def start_ingestion(website_id: str, max_pages: int = 50):
    """Start ingestion for a website."""
    # Get website details
    website = get_website(website_id)
    if not website:
        raise HTTPException(status_code=404, detail="Website not found")
    
    # Check if already ingesting
    if website.status == "ingesting":
        raise HTTPException(status_code=400, detail="Ingestion already in progress")
    
    # Create ingestion job
    job_data = IngestionJobCreate(website_id=website_id)
    job = create_ingestion_job(job_data)
    if not job:
        raise HTTPException(status_code=500, detail="Failed to create ingestion job")
    
    # Start background job
    success = job_manager.start_job(
        job_id=job.id,
        target=run_ingestion_job,
        kwargs={
            "website_id": website_id,
            "job_id": job.id,
            "domain": website.url,
            "namespace": website.namespace,
            "max_pages": max_pages
        }
    )
    
    if not success:
        raise HTTPException(status_code=500, detail="Failed to start ingestion job")
    
    return {
        "message": "Ingestion started",
        "job_id": job.id,
        "website_id": website_id,
        "namespace": website.namespace
    }


@app.get("/websites/{website_id}/jobs", response_model=List[IngestionJobResponse])
async def get_website_jobs(website_id: str):
    """Get ingestion jobs for a website."""
    return list_ingestion_jobs(website_id)


@app.get("/jobs/{job_id}", response_model=IngestionJobResponse)
async def get_job_detail(job_id: str):
    """Get ingestion job details."""
    job = get_ingestion_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@app.get("/jobs/{job_id}/status")
async def get_job_runtime_status(job_id: str):
    """Get real-time job status from job manager."""
    job = get_ingestion_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    # Get runtime status from job manager
    runtime_status = job_manager.get_job_status(job_id)
    
    return {
        "job_id": job_id,
        "db_status": job.status,
        "progress_percent": job.progress_percent,
        "pages_crawled": job.pages_crawled,
        "chunks_created": job.chunks_created,
        "runtime_status": runtime_status,
        "started_at": job.started_at,
        "completed_at": job.completed_at,
        "error_message": job.error_message
    }


@app.exception_handler(422)
async def validation_error_handler(request: Request, exc):
    return {
        "error": "Validation error",
        "detail": "Please check your request format",
        "example": {
            "question": "What services does Zibtek offer?",
            "session_id": "optional-session-id"
        }
    }


# ===== AUTHENTICATION ENDPOINTS =====

async def get_current_user(request: Request) -> UserResponse:
    """Extract and verify user from JWT token in Authorization header."""
    auth_header = request.headers.get("Authorization")
    
    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(
            status_code=401,
            detail="Missing or invalid authorization header"
        )
    
    token = auth_header.replace("Bearer ", "")
    payload = verify_access_token(token)
    
    if not payload:
        raise HTTPException(
            status_code=401,
            detail="Invalid or expired token"
        )
    
    user = get_user_by_id(payload.get("user_id"))
    if not user:
        raise HTTPException(
            status_code=401,
            detail="User not found"
        )
    
    return user


@app.post("/auth/signup", response_model=AuthResponse)
async def signup(user_data: UserSignup):
    """Create a new user account."""
    user = create_user(user_data)
    
    if not user:
        raise HTTPException(
            status_code=400,
            detail="User with this email already exists or registration failed"
        )
    
    # Generate JWT token
    token, expires_at = create_access_token(user.id, user.email)
    
    return AuthResponse(
        user=user,
        token=token,
        expires_at=expires_at
    )


@app.post("/auth/login", response_model=AuthResponse)
async def login(login_data: UserLogin):
    """Authenticate user and return JWT token."""
    user = authenticate_user(login_data)
    
    if not user:
        raise HTTPException(
            status_code=401,
            detail="Invalid email or password"
        )
    
    # Generate JWT token
    token, expires_at = create_access_token(user.id, user.email)
    
    return AuthResponse(
        user=user,
        token=token,
        expires_at=expires_at
    )


@app.get("/auth/me", response_model=UserResponse)
async def get_current_user_info(request: Request):
    """Get current authenticated user information."""
    user = await get_current_user(request)
    return user


# ===== CONVERSATION HISTORY ENDPOINTS =====

@app.get("/conversations")
async def get_conversations(request: Request, namespace: Optional[str] = None):
    """Get all conversation sessions for the authenticated user."""
    user = await get_current_user(request)
    conversations = get_user_conversations(user.id, namespace)
    return {"conversations": conversations}


@app.get("/conversations/{namespace}/{session_id}/messages")
async def get_messages(request: Request, namespace: str, session_id: str):
    """Get all messages in a specific conversation."""
    user = await get_current_user(request)
    messages = get_conversation_messages(user.id, session_id, namespace)
    return {"messages": messages}


@app.delete("/conversations/{namespace}/{session_id}")
async def delete_conversation_endpoint(request: Request, namespace: str, session_id: str):
    """Delete a conversation and all its messages."""
    user = await get_current_user(request)
    success = delete_conversation(user.id, session_id, namespace)
    
    if not success:
        raise HTTPException(
            status_code=500,
            detail="Failed to delete conversation"
        )
    
    return {"message": "Conversation deleted successfully"}


@app.get("/hybrid/stats")
async def get_hybrid_stats():
    """Get statistics about cached hybrid retrievers.
    
    Returns info about which namespaces have active retrievers,
    their age, and memory usage.
    """
    if not hybrid_manager:
        return {
            "enabled": False,
            "message": "Hybrid search is not enabled"
        }
    
    stats = hybrid_manager.get_stats()
    return {
        "enabled": True,
        "manager_stats": stats
    }


@app.post("/hybrid/clear/{namespace}")
async def clear_hybrid_cache(namespace: str):
    """Clear cached hybrid retriever for a specific namespace.
    
    Useful after re-ingestion to force rebuild of BM25 index.
    """
    if not hybrid_manager:
        raise HTTPException(
            status_code=400,
            detail="Hybrid search is not enabled"
        )
    
    cleared = hybrid_manager.clear_namespace(namespace)
    
    if cleared:
        return {
            "message": f"Cleared hybrid retriever cache for namespace: {namespace}",
            "namespace": namespace
        }
    else:
        return {
            "message": f"No cached retriever found for namespace: {namespace}",
            "namespace": namespace
        }


@app.post("/hybrid/clear-all")
async def clear_all_hybrid_caches():
    """Clear all cached hybrid retrievers.
    
    Forces rebuild of all BM25 indices on next use.
    """
    if not hybrid_manager:
        raise HTTPException(
            status_code=400,
            detail="Hybrid search is not enabled"
        )
    
    hybrid_manager.clear_all()
    
    return {
        "message": "Cleared all hybrid retriever caches",
        "note": "Retrievers will be rebuilt on-demand per namespace"
    }


# ===== DOCUMENT UPLOAD ENDPOINT =====

@app.post("/documents/upload")
async def upload_document(
    request: Request,
    file: UploadFile = File(...),
    org_id: str = Form(...)
):
    """Upload and ingest a document (PDF, TXT, DOCX) into an organization's namespace.
    
    Args:
        file: The uploaded file
        org_id: Organization ID (namespace will be auto-generated from org slug)
        
    Returns:
        Ingestion result with status and metadata
    """
    user = await get_current_user(request)
    
    # Import here to avoid startup dependency
    from ingest.document_ingest import DocumentIngestor
    
    try:
        # Validate file format
        if not DocumentIngestor.is_supported_format(file.filename):
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file format. Supported: {', '.join(DocumentIngestor.get_supported_formats())}"
            )
        
        # Check file size (max 10MB)
        file_content = await file.read()
        file_size_mb = len(file_content) / (1024 * 1024)
        
        if file_size_mb > 10:
            raise HTTPException(
                status_code=400,
                detail=f"File too large ({file_size_mb:.1f}MB). Maximum size is 10MB"
            )
        
        # Get organization to generate namespace
        org = get_organization(org_id)
        if not org:
            raise HTTPException(status_code=404, detail="Organization not found")
        
        # Generate namespace from organization slug
        from storage.organizations import create_namespace
        namespace = create_namespace(org.slug)
        
        print(f"📤 Upload request from user {user.email}")
        print(f"   File: {file.filename} ({file_size_mb:.2f}MB)")
        print(f"   Organization: {org.name} (slug: {org.slug})")
        print(f"   Namespace: {namespace}")
        
        # Initialize ingestor
        ingestor = DocumentIngestor(namespace=namespace)
        
        # Ingest document
        result = ingestor.ingest_document(
            file_content=file_content,
            filename=file.filename,
            org_id=org_id,
            user_id=user.id,
            metadata={
                'uploaded_by': user.id,
                'uploaded_by_email': user.email
            }
        )
        
        if result['success']:
            return {
                "success": True,
                "message": f"Successfully ingested {file.filename}",
                "doc_id": result['doc_id'],
                "chunks": result['chunks'],
                "characters": result['characters'],
                "namespace": namespace,
                "elapsed_seconds": result['elapsed_seconds']
            }
        else:
            raise HTTPException(
                status_code=500,
                detail=result.get('error', 'Unknown error during ingestion')
            )
            
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error uploading document: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing document: {str(e)}"
        )


# ===== DOCUMENT MANAGEMENT ENDPOINTS =====

@app.get("/documents/org/{org_id}")
async def get_org_documents(request: Request, org_id: str):
    """Get all documents for an organization."""
    user = await get_current_user(request)
    documents = get_documents_by_org(org_id)
    return {"documents": documents}


@app.get("/documents/namespace/{namespace}")
async def get_namespace_documents(request: Request, namespace: str):
    """Get all documents in a namespace."""
    user = await get_current_user(request)
    documents = get_documents_by_namespace(namespace)
    return {"documents": documents}


@app.get("/documents/stats")
async def get_docs_stats(request: Request, org_id: Optional[str] = None):
    """Get document statistics."""
    user = await get_current_user(request)
    stats = get_document_stats(org_id)
    return stats


@app.delete("/documents/{doc_id}")
async def delete_document(request: Request, doc_id: str):
    """Delete a document record (vectors must be deleted separately from Pinecone)."""
    user = await get_current_user(request)
    success = delete_document_record(doc_id)
    
    if not success:
        raise HTTPException(
            status_code=500,
            detail="Failed to delete document"
        )
    
    return {"success": True, "message": f"Document {doc_id} deleted"}


if __name__ == "__main__":
    import uvicorn
    
    print("🚀 Starting Zibtek Chatbot Server...")
    print("📍 Server will be available at: http://localhost:8000")
    print("📖 API docs available at: http://localhost:8000/docs")
    
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )