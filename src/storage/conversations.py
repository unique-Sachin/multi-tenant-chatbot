"""Conversation history management per user and namespace."""

from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel

from .db import supabase


class ConversationMessage(BaseModel):
    """Model for a conversation message."""
    id: str
    user_id: str
    namespace: str
    session_id: str
    user_query: str
    answer: str
    citations: List[str]
    is_out_of_scope: bool
    processing_time_ms: int
    retrieval_steps: Dict[str, Any]
    created_at: datetime


class ConversationSession(BaseModel):
    """Model for a conversation session."""
    id: str
    user_id: str
    namespace: str
    session_id: str
    title: Optional[str]
    message_count: int
    created_at: datetime
    updated_at: datetime


def get_user_conversations(user_id: str, namespace: Optional[str] = None) -> List[ConversationSession]:
    """Get all conversation sessions for a user, optionally filtered by namespace."""
    if not supabase:
        print("❌ Supabase client not initialized")
        return []
    
    try:
        query = supabase.table("conversation_sessions").select("*").eq("user_id", user_id)
        
        if namespace:
            query = query.eq("namespace", namespace)
        
        result = query.order("updated_at", desc=True).execute()
        
        if result.data:
            sessions = []
            for session_data in result.data:
                # Count messages in this session
                msg_count = supabase.table("chat_logs")\
                    .select("id", count="exact")\
                    .eq("user_id", user_id)\
                    .eq("session_id", session_data['session_id'])\
                    .execute()
                
                session_data['message_count'] = len(msg_count.data) if msg_count.data else 0
                sessions.append(ConversationSession(**session_data))
            
            return sessions
        return []
        
    except Exception as e:
        print(f"❌ Error getting conversations: {e}")
        return []


def get_conversation_messages(user_id: str, session_id: str, namespace: str) -> List[ConversationMessage]:
    """Get all messages for a specific conversation session."""
    if not supabase:
        print("❌ Supabase client not initialized")
        return []
    
    try:
        result = supabase.table("chat_logs")\
            .select("*")\
            .eq("user_id", user_id)\
            .eq("session_id", session_id)\
            .eq("namespace", namespace)\
            .order("ts", desc=False)\
            .execute()
        
        if result.data:
            messages = []
            for msg_data in result.data:
                # Convert citations from DB format
                citations = []
                if msg_data.get('citations'):
                    citations = [c.get('url', '') for c in msg_data['citations'] if isinstance(c, dict)]
                elif msg_data.get('retrieved_urls'):
                    citations = msg_data['retrieved_urls']
                
                messages.append(ConversationMessage(
                    id=msg_data['id'],
                    user_id=msg_data['user_id'],
                    namespace=msg_data.get('namespace', 'zibtek'),
                    session_id=msg_data['session_id'],
                    user_query=msg_data['user_query'],
                    answer=msg_data['answer'],
                    citations=citations,
                    is_out_of_scope=msg_data.get('is_oos', False),
                    processing_time_ms=msg_data.get('latency_ms', 0),
                    retrieval_steps=msg_data.get('retrieval_steps', {}),
                    created_at=datetime.fromisoformat(msg_data['ts'].replace('Z', '+00:00'))
                ))
            
            return messages
        return []
        
    except Exception as e:
        print(f"❌ Error getting conversation messages: {e}")
        return []


def create_or_update_session(user_id: str, session_id: str, namespace: str, title: Optional[str] = None) -> bool:
    """Create or update a conversation session."""
    if not supabase:
        print("❌ Supabase client not initialized")
        return False
    
    try:
        # Check if session exists
        existing = supabase.table("conversation_sessions")\
            .select("id")\
            .eq("user_id", user_id)\
            .eq("session_id", session_id)\
            .eq("namespace", namespace)\
            .execute()
        
        if existing.data and len(existing.data) > 0:
            # Update existing session
            supabase.table("conversation_sessions")\
                .update({"updated_at": datetime.utcnow().isoformat()})\
                .eq("user_id", user_id)\
                .eq("session_id", session_id)\
                .eq("namespace", namespace)\
                .execute()
        else:
            # Create new session
            session_data = {
                "user_id": user_id,
                "session_id": session_id,
                "namespace": namespace,
                "title": title or f"Conversation on {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                "created_at": datetime.utcnow().isoformat(),
                "updated_at": datetime.utcnow().isoformat()
            }
            supabase.table("conversation_sessions").insert(session_data).execute()
        
        return True
        
    except Exception as e:
        print(f"❌ Error creating/updating session: {e}")
        return False


def delete_conversation(user_id: str, session_id: str, namespace: str) -> bool:
    """Delete a conversation session and all its messages."""
    if not supabase:
        print("❌ Supabase client not initialized")
        return False
    
    try:
        # Delete messages
        supabase.table("chat_logs")\
            .delete()\
            .eq("user_id", user_id)\
            .eq("session_id", session_id)\
            .eq("namespace", namespace)\
            .execute()
        
        # Delete session
        supabase.table("conversation_sessions")\
            .delete()\
            .eq("user_id", user_id)\
            .eq("session_id", session_id)\
            .eq("namespace", namespace)\
            .execute()
        
        print(f"✅ Deleted conversation: {session_id}")
        return True
        
    except Exception as e:
        print(f"❌ Error deleting conversation: {e}")
        return False
