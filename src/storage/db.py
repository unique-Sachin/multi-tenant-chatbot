"""Database configuration and models for the Zibtek chatbot using Supabase."""

import os
import uuid
from datetime import datetime
from typing import Optional, Dict, Any, List

from supabase import create_client, Client
from pydantic import BaseModel
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Supabase configuration
url = os.environ.get("SUPABASE_URL")
key = os.environ.get("SUPABASE_ANON_KEY")

print(f"Supabase URL: {'Set' if url else 'Not Set'}")

# Create Supabase client only if credentials are provided
supabase: Optional[Client] = None
if url and key:
    try:
        supabase = create_client(url, key)
        print("✅ Supabase client initialized successfully")
    except Exception as e:
        print(f"⚠️  Supabase client initialization failed: {e}")
        supabase = None
else:
    print("⚠️  Supabase credentials not found. Set SUPABASE_URL and SUPABASE_ANON_KEY in .env")

# Pydantic models for API
class ChatLogCreate(BaseModel):
    """Pydantic model for creating chat logs."""
    session_id: str
    user_id: Optional[str] = None  # User ID for authentication
    namespace: Optional[str] = "zibtek"  # Organization namespace
    user_query: str
    is_oos: bool = False
    retrieved_ids: Optional[List[str]] = None
    retrieved_urls: Optional[List[str]] = None
    rerank_scores: Optional[Dict[str, float]] = None
    answer: str
    citations: Optional[List[Dict[str, Any]]] = None
    model: Optional[str] = None
    latency_ms: Optional[int] = None
    cost_cents: Optional[int] = None
    flags: Optional[Dict[str, Any]] = None
    retrieval_steps: Optional[Dict[str, Any]] = None


class ChatLogResponse(BaseModel):
    """Pydantic model for chat log responses."""
    id: str
    ts: datetime
    session_id: str
    user_query: str
    is_oos: bool
    retrieved_ids: Optional[List[str]]
    retrieved_urls: Optional[List[str]]
    rerank_scores: Optional[Dict[str, float]]
    answer: str
    citations: Optional[List[Dict[str, Any]]]
    model: Optional[str]
    latency_ms: Optional[int]
    cost_cents: Optional[int]
    flags: Optional[Dict[str, Any]]
    
    class Config:
        from_attributes = True


# Supabase database functions
def create_chat_log(chat_log: ChatLogCreate) -> Optional[Dict[str, Any]]:
    """Create a new chat log entry using Supabase client."""
    if not supabase:
        print("❌ Supabase client not initialized. Set SUPABASE_URL and SUPABASE_ANON_KEY.")
        return None
        
    chat_data = {
        "session_id": chat_log.session_id,
        "user_id": chat_log.user_id,
        "namespace": chat_log.namespace,
        "user_query": chat_log.user_query,
        "is_oos": chat_log.is_oos,
        "retrieved_ids": chat_log.retrieved_ids,
        "retrieved_urls": chat_log.retrieved_urls,
        "rerank_scores": chat_log.rerank_scores,
        "answer": chat_log.answer,
        "citations": chat_log.citations,
        "model": chat_log.model,
        "latency_ms": chat_log.latency_ms,
        "cost_cents": chat_log.cost_cents,
        "flags": chat_log.flags,
        "retrieval_steps": chat_log.retrieval_steps,
    }
    
    try:
        result = supabase.table("chat_logs").insert(chat_data).execute()
        # Handle Supabase response
        if hasattr(result, 'data') and result.data:
            return result.data[0]  # type: ignore
        return None
    except Exception as e:
        print(f"Error creating chat log: {e}")
        return None


def get_chat_logs(
    session_id: Optional[str] = None,
    is_oos: Optional[bool] = None,
    limit: int = 100,
    offset: int = 0
) -> List[Dict[str, Any]]:
    """Retrieve chat logs using Supabase client."""
    if not supabase:
        print("❌ Supabase client not initialized. Set SUPABASE_URL and SUPABASE_ANON_KEY.")
        return []
        
    try:
        query = supabase.table("chat_logs").select("*")
        
        if session_id:
            query = query.eq("session_id", session_id)
        
        if is_oos is not None:
            query = query.eq("is_oos", is_oos)
        
        result = query.order("ts", desc=True).range(offset, offset + limit - 1).execute()
        # Handle Supabase response
        if hasattr(result, 'data') and result.data:
            return result.data  # type: ignore
        return []
    except Exception as e:
        print(f"Error retrieving chat logs: {e}")
        return []


def get_chat_log_stats() -> Dict[str, Any]:
    """Get basic statistics about chat logs using Supabase client."""
    if not supabase:
        print("❌ Supabase client not initialized. Set SUPABASE_URL and SUPABASE_ANON_KEY.")
        return {
            "total_logs": 0,
            "in_scope_logs": 0,
            "out_of_scope_logs": 0,
            "average_latency_ms": None,
            "out_of_scope_percentage": 0
        }
        
    try:
        # Get total count and out-of-scope count
        all_logs = supabase.table("chat_logs").select("is_oos, latency_ms").execute()
        
        if not hasattr(all_logs, 'data') or not all_logs.data:
            return {
                "total_logs": 0,
                "in_scope_logs": 0,
                "out_of_scope_logs": 0,
                "average_latency_ms": None,
                "out_of_scope_percentage": 0
            }
        
        logs_data = all_logs.data  # type: ignore
        total_logs = len(logs_data)
        oos_logs = sum(1 for log in logs_data if isinstance(log, dict) and log.get("is_oos", False))
        in_scope_logs = total_logs - oos_logs
        
        # Calculate average latency
        latencies = [
            log.get("latency_ms") for log in logs_data 
            if isinstance(log, dict) and log.get("latency_ms") is not None 
            and isinstance(log.get("latency_ms"), (int, float))
        ]
        avg_latency = round(sum(latencies) / len(latencies), 2) if latencies else None  # type: ignore
        
        return {
            "total_logs": total_logs,
            "in_scope_logs": in_scope_logs,
            "out_of_scope_logs": oos_logs,
            "average_latency_ms": avg_latency,
            "out_of_scope_percentage": round((oos_logs / total_logs) * 100, 2) if total_logs > 0 else 0
        }
    except Exception as e:
        print(f"Error getting chat log stats: {e}")
        return {
            "total_logs": 0,
            "in_scope_logs": 0,
            "out_of_scope_logs": 0,
            "average_latency_ms": None,
            "out_of_scope_percentage": 0
        }