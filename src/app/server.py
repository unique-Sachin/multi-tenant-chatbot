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
from fastapi.middleware.cors import CORSMiddleware
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
from retrieval.retriever import MilvusRetriever  # Changed from ZibtekRetriever
from retrieval.rerank import rerank_with_scores
from storage.db import create_chat_log, ChatLogCreate
from storage.auth import (
    UserSignup, UserLogin, UserResponse, AuthResponse,
    create_user, authenticate_user, get_user_by_id,
    create_access_token, verify_access_token
)
from storage.conversations import (
    get_user_conversations, get_conversation_messages,
    create_or_update_session, delete_conversation,
    get_last_conversation_message, get_last_conversation_messages
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
    partition_name: Optional[str] = Field("_default", description="Organization partition for multi-tenant retrieval")


class ChatStreamRequest(BaseModel):
    """Chat streaming request model."""
    question: str = Field(..., min_length=1, max_length=1000, description="User's question")
    session_id: Optional[str] = Field(None, description="Optional session identifier")
    partition_name: Optional[str] = Field("_default", description="Organization partition for multi-tenant retrieval")


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
retriever: Optional[MilvusRetriever] = None  # Changed to MilvusRetriever
llm: Optional[ChatOpenAI] = None
tokenizer: Optional[tiktoken.Encoding] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize services on startup."""
    global guards, retriever, llm, tokenizer, cache_manager
    
    print("🚀 Starting Zibtek Chatbot Server...")
    
    try:
        # Initialize cache manager
        if CACHE_ENABLED:
            print("💾 Initializing cache manager...")
            cache_manager = create_cache_manager()
        
        # Initialize guardrails
        print("🛡️  Initializing guardrails...")
        guards = ZibtekGuards()
        
        # Initialize Milvus retriever (with built-in hybrid search)
        print("� Initializing Milvus retriever (hybrid: dense + BM25)...")
        retriever = MilvusRetriever()
        print("✅ Milvus retriever ready with built-in hybrid search and multi-tenant partitions")
        
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

# Add CORS middleware to allow all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
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
    
    gpt-4o-mini pricing (as of 2024):
    - Input: $0.01 per 1K tokens
    - Output: $0.03 per 1K tokens
    """
    input_cost = (input_tokens / 1000) * 0.01
    output_cost = (output_tokens / 1000) * 0.03
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
    partition_name: str,
    rerank_scores: Optional[Dict[str, float]] = None
) -> None:
    """Log chat interaction to database."""
    try:
        # Alias partition_name as namespace for backward compatibility
        namespace = partition_name
        
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
            
            # Get partition name from request (multi-tenant support)
            partition_name = request.partition_name or "_default"
            namespace = partition_name  # Alias for backward compatibility with other functions
            
            # Pass partition_name to guardrails for multi-tenant scope checking
            is_oos, reason = guards.is_out_of_scope(request.question, partition_name)
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
            
            # Milvus hybrid search (built-in: dense + BM25 + RRF)
            print(f"🔀 Using Milvus hybrid retrieval (dense + BM25 + RRF) for partition: '{partition_name}'...")
            
            # Single call for hybrid search with partition-based multi-tenancy
            docs_with_scores = retriever.hybrid_search(
                query=request.question,
                k=20,
                partition_name=partition_name  # Multi-tenant isolation
            )
            
            # Extract documents and scores
            docs = [doc for doc, score in docs_with_scores]
            scores = {f"doc_{i}": score for i, (doc, score) in enumerate(docs_with_scores[:5])}
            
            # Capture retrieval step information
            retrieval_steps["hybrid_search"] = {
                "count": len(docs),
                "scores": scores,
                "method": "milvus_hybrid (dense + BM25 + RRF)",
                "partition": partition_name
            }
            retrieval_steps["method"] = "milvus_hybrid"
            
            print(f"✅ Milvus hybrid retrieval complete: {len(docs)} documents from partition '{partition_name}'")
            
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
            
            # Get conversation history for context (if session exists)
            conversation_history = ""
            if request.session_id:
                print(f"💬 Checking for conversation history in session: {request.session_id}")
                last_messages = get_last_conversation_messages(user.id, request.session_id, namespace, limit=3)
                
                # Filter out out-of-scope messages and build history
                valid_messages = [msg for msg in last_messages if not msg.is_out_of_scope]
                
                if valid_messages:
                    history_parts = []
                    for i, msg in enumerate(valid_messages, 1):
                        history_parts.append(f"Previous conversation {i}:")
                        history_parts.append(f"User's question: {msg.user_query}")
                        history_parts.append(f"Your answer: {msg.answer}")
                        history_parts.append("")  # Empty line for separation
                    
                    conversation_history = "\n".join(history_parts)
                    print(f"✅ Found {len(valid_messages)} previous conversations - will be included as context")
                else:
                    print(f"📭 No previous valid conversations found in this session")
            
            # Prepare context and messages
            context = "\n\n".join([doc.page_content for doc in filtered_docs])
            citations = extract_citations(filtered_docs)  # Use the proper citation extraction function
            
            # Get organization info for dynamic system prompt
            from storage.organizations import get_organization_by_namespace
            org_info = get_organization_by_namespace(namespace)
            
            system_prompt = guards.system_prompt(org_info)
            
            # Build the user prompt with conversation history if available
            user_prompt_parts = []
            
            if conversation_history:
                user_prompt_parts.append(conversation_history)
            
            user_prompt_parts.extend([
                f"Current question: {request.question}",
                "",
                f"Context:\n{context}",
                "",
                "Please answer the current question using ONLY the information provided in the CONTEXT above. If this is a follow-up question, consider the previous conversation for context but base your answer on the provided CONTEXT. Always cite the URL(s) where you found the information."
            ])
            
            user_prompt = "\n".join(user_prompt_parts)
            
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
        "available_endpoints": ["/", "/health", "/chat/stream", "/stats"]
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
async def start_ingestion(website_id: str, max_pages: int = 500):
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
    
    user_id = payload.get("user_id")
    if not user_id:
        raise HTTPException(
            status_code=401,
            detail="Invalid token: missing user_id"
        )
    
    user = get_user_by_id(user_id)
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


# Hybrid endpoints removed - Milvus has built-in hybrid search








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
        if not file.filename:
            raise HTTPException(
                status_code=400,
                detail="No filename provided"
            )
        
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
        ingestor = DocumentIngestor(partition_name=namespace)
        
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