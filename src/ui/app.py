"""Streamlit chat UI with user authentication and conversation history."""

import streamlit as st
import requests
import json
import uuid
import time
from datetime import datetime
from typing import List, Dict, Any, Optional
import os
from dotenv import load_dotenv

load_dotenv()

# Configuration
API_BASE_URL = os.getenv("API_BASE_URL", "https://zibtek-chatbot-747d3d71730d.herokuapp.com")

# Debug: Show which API we're connecting to
print(f"🌐 Connecting to API: {API_BASE_URL}")


def initialize_session():
    """Initialize session state variables."""
    if "user" not in st.session_state:
        st.session_state.user = None
    
    if "token" not in st.session_state:
        st.session_state.token = None
    
    if "current_namespace" not in st.session_state:
        st.session_state.current_namespace = "zibtek"
    
    if "current_session_id" not in st.session_state:
        st.session_state.current_session_id = str(uuid.uuid4())
    
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    if "conversations" not in st.session_state:
        st.session_state.conversations = {}
    
    if "organizations" not in st.session_state:
        st.session_state.organizations = []
    
    if "websites" not in st.session_state:
        st.session_state.websites = []


def call_api_with_auth(method: str, endpoint: str, data: Optional[Dict] = None) -> Optional[Dict]:
    """Make API call with authentication token."""
    headers = {}
    if st.session_state.token:
        headers["Authorization"] = f"Bearer {st.session_state.token}"
    
    url = f"{API_BASE_URL}{endpoint}"
    
    try:
        if method == "GET":
            response = requests.get(url, headers=headers, timeout=30)
        elif method == "POST":
            response = requests.post(url, json=data, headers=headers, timeout=30)
        elif method == "DELETE":
            response = requests.delete(url, headers=headers, timeout=30)
        else:
            return None
        
        if response.status_code == 401:
            # Token expired or invalid
            st.session_state.user = None
            st.session_state.token = None
            st.error("Session expired. Please log in again.")
            st.rerun()
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        st.error(f"Connection error: {str(e)}")
        return None


def login_user(email: str, password: str) -> bool:
    """Login user and store token."""
    data = {"email": email, "password": password}
    response = requests.post(f"{API_BASE_URL}/auth/login", json=data)
    
    if response.status_code == 200:
        result = response.json()
        st.session_state.user = result["user"]
        st.session_state.token = result["token"]
        return True
    else:
        return False


def signup_user(email: str, password: str, full_name: Optional[str] = None) -> bool:
    """Signup new user and store token."""
    data = {"email": email, "password": password, "full_name": full_name}
    response = requests.post(f"{API_BASE_URL}/auth/signup", json=data)
    
    if response.status_code == 200:
        result = response.json()
        st.session_state.user = result["user"]
        st.session_state.token = result["token"]
        return True
    else:
        return False


def logout_user():
    """Logout user and clear session."""
    st.session_state.user = None
    st.session_state.token = None
    st.session_state.chat_history = []
    st.session_state.conversations = {}
    st.session_state.current_session_id = str(uuid.uuid4())
    st.rerun()


def fetch_conversations():
    """Fetch user's conversations."""
    result = call_api_with_auth("GET", "/conversations")
    if result:
        conversations = result.get("conversations", [])
        # Group by namespace
        grouped = {}
        for conv in conversations:
            namespace = conv.get("namespace", "zibtek")
            if namespace not in grouped:
                grouped[namespace] = []
            grouped[namespace].append(conv)
        st.session_state.conversations = grouped


def fetch_conversation_messages(namespace: str, session_id: str):
    """Fetch messages for a specific conversation."""
    result = call_api_with_auth("GET", f"/conversations/{namespace}/{session_id}/messages")
    if result:
        messages = result.get("messages", [])
        # Convert to chat history format
        chat_history = []
        for msg in messages:
            # User message
            chat_history.append({
                "content": msg.get("user_query", ""),
                "is_user": True,
                "timestamp": msg.get("created_at", datetime.now().isoformat())
            })
            # Bot response
            chat_history.append({
                "content": msg.get("answer", ""),
                "citations": msg.get("citations", []),
                "retrieval_steps": msg.get("retrieval_steps", {}),
                "is_user": False,
                "is_out_of_scope": msg.get("is_out_of_scope", False),
                "processing_time_ms": msg.get("processing_time_ms", msg.get("latency_ms", 0)),
                "timestamp": msg.get("created_at", datetime.now().isoformat())
            })
        
        st.session_state.chat_history = chat_history
        st.session_state.current_session_id = session_id
        st.session_state.current_namespace = namespace
        return True
    return False


def delete_conversation(namespace: str, session_id: str):
    """Delete a conversation."""
    result = call_api_with_auth("DELETE", f"/conversations/{namespace}/{session_id}")
    if result:
        fetch_conversations()
        if st.session_state.current_session_id == session_id:
            st.session_state.current_session_id = str(uuid.uuid4())
            st.session_state.chat_history = []


def fetch_organizations():
    """Fetch all organizations."""
    result = call_api_with_auth("GET", "/organizations")
    if result:
        st.session_state.organizations = result


def fetch_websites():
    """Fetch all websites."""
    result = call_api_with_auth("GET", "/websites")
    if result:
        st.session_state.websites = result


def send_message(question: str, namespace: str) -> Optional[Dict]:
    """Send message to chat API."""
    data = {
        "question": question,
        "session_id": st.session_state.current_session_id,
        "namespace": namespace
    }
    return call_api_with_auth("POST", "/chat", data)


def render_auth_page():
    """Render login/signup page."""
    st.markdown("<h1 style='text-align: center;'>🤖 Multi-Tenant AI Assistant</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #666;'>Please log in or create an account to continue</p>", unsafe_allow_html=True)
    
    tab_login, tab_signup = st.tabs(["🔑 Login", "📝 Sign Up"])
    
    with tab_login:
        st.subheader("Login to your account")
        
        with st.form("login_form"):
            email = st.text_input("Email", placeholder="your@email.com")
            password = st.text_input("Password", type="password")
            submit = st.form_submit_button("Login", type="primary", use_container_width=True)
            
            if submit:
                if not email or not password:
                    st.error("Please enter both email and password")
                else:
                    with st.spinner("Logging in..."):
                        if login_user(email, password):
                            st.success("✅ Login successful!")
                            time.sleep(0.5)
                            st.rerun()
                        else:
                            st.error("❌ Invalid email or password")
    
    with tab_signup:
        st.subheader("Create a new account")
        
        with st.form("signup_form"):
            full_name = st.text_input("Full Name (optional)", placeholder="John Doe")
            email = st.text_input("Email", placeholder="your@email.com")
            password = st.text_input("Password", type="password", help="Minimum 6 characters")
            password_confirm = st.text_input("Confirm Password", type="password")
            submit = st.form_submit_button("Sign Up", type="primary", use_container_width=True)
            
            if submit:
                if not email or not password:
                    st.error("Please enter email and password")
                elif len(password) < 6:
                    st.error("Password must be at least 6 characters")
                elif password != password_confirm:
                    st.error("Passwords do not match")
                else:
                    with st.spinner("Creating account..."):
                        if signup_user(email, password, full_name or None):
                            st.success("✅ Account created successfully!")
                            time.sleep(0.5)
                            st.rerun()
                        else:
                            st.error("❌ Failed to create account. Email may already be in use.")


def render_retrieval_steps(retrieval_steps: Dict[str, Any]):
    """Render detailed retrieval pipeline steps."""
    if not retrieval_steps or not any(retrieval_steps.values()):
        return
    
    with st.expander("🔍 **Retrieval Pipeline Details** (click to expand)", expanded=False):
        method = retrieval_steps.get("method", "unknown")
        st.markdown(f"**Method:** `{method}`")
        st.markdown("---")
        
        # Vector Search
        vector_info = retrieval_steps.get("vector_search", {})
        if vector_info:
            st.markdown("### 1️⃣ Vector Search (Semantic)")
            st.write(f"**Retrieved:** {vector_info.get('count', 0)} documents")
            
            if vector_info.get('scores'):
                st.write("**Top Scores:**")
                scores_data = []
                for doc_id, score in list(vector_info['scores'].items())[:3]:
                    scores_data.append({"Document ID": doc_id[:50], "Similarity": f"{score:.4f}"})
                if scores_data:
                    st.dataframe(scores_data, use_container_width=True, hide_index=True)
        
        # BM25 Search
        bm25_info = retrieval_steps.get("bm25_search", {})
        if bm25_info and bm25_info.get('count', 0) > 0:
            st.markdown("### 2️⃣ BM25 Search (Keyword)")
            st.write(f"**Retrieved:** {bm25_info.get('count', 0)} documents")
            
            if bm25_info.get('scores'):
                st.write("**Top Scores:**")
                scores_data = []
                for doc_id, score in list(bm25_info['scores'].items())[:3]:
                    scores_data.append({"Document ID": doc_id[:50], "BM25 Score": f"{score:.4f}"})
                if scores_data:
                    st.dataframe(scores_data, use_container_width=True, hide_index=True)
        
        # RRF Fusion
        rrf_info = retrieval_steps.get("rrf_fusion", {})
        if rrf_info and rrf_info.get('count', 0) > 0:
            st.markdown("### 3️⃣ RRF Fusion (Reciprocal Rank)")
            st.write(f"**Fused:** {rrf_info.get('count', 0)} documents")
            
            if rrf_info.get('scores'):
                st.write("**Top Fusion Scores:**")
                scores_data = []
                for doc_id, score in list(rrf_info['scores'].items())[:3]:
                    scores_data.append({"Document ID": doc_id[:50], "RRF Score": f"{score:.4f}"})
                if scores_data:
                    st.dataframe(scores_data, use_container_width=True, hide_index=True)
        
        # Reranking
        rerank_info = retrieval_steps.get("reranking", {})
        if rerank_info:
            st.markdown("### 4️⃣ Reranking (Cohere)")
            
            if rerank_info.get('enabled'):
                if rerank_info.get('error'):
                    st.error(f"❌ Error: {rerank_info['error']}")
                else:
                    st.write(f"**Input:** {rerank_info.get('input_count', 0)} documents")
                    st.write(f"**Output:** {rerank_info.get('output_count', 0)} documents")
                    
                    if rerank_info.get('scores'):
                        st.write("**Top Relevance Scores:**")
                        scores_data = []
                        for doc_id, score in list(rerank_info['scores'].items())[:3]:
                            scores_data.append({"Document ID": doc_id[:50], "Relevance": f"{score:.4f}"})
                        if scores_data:
                            st.dataframe(scores_data, use_container_width=True, hide_index=True)
            else:
                reason = rerank_info.get('reason', 'unknown')
                st.info(f"⚠️ Skipped: {reason}")


def render_citations(citations: List[str]):
    """Render citations as clickable links."""
    if not citations:
        return
    
    st.write("**📚 Sources:**")
    for i, url in enumerate(citations, 1):
        from urllib.parse import urlparse
        domain = urlparse(url).netloc or url
        display_name = domain.replace('www.', '')
        
        st.markdown(
            f'<a href="{url}" target="_blank" style="'
            f'display: inline-block; '
            f'margin: 4px; '
            f'padding: 6px 12px; '
            f'background-color: #f0f2f6; '
            f'border-radius: 4px; '
            f'text-decoration: none; '
            f'color: #1f77b4; '
            f'border: 1px solid #d1d5db; '
            f'font-size: 0.8rem; '
            f'font-weight: 500;">'
            f'🔗 {display_name}</a>',
            unsafe_allow_html=True
        )


def render_message(message: Dict[str, Any], is_user: bool = False):
    """Render a chat message."""
    if is_user:
        with st.chat_message("user"):
            st.write(message["content"])
    else:
        with st.chat_message("assistant"):
            st.write(message["content"])
            
            if message.get("retrieval_steps"):
                render_retrieval_steps(message["retrieval_steps"])
            
            if message.get("citations"):
                render_citations(message["citations"])
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                processing_time = message.get("processing_time_ms", 0)
                st.caption(f"⏱️ {processing_time}ms")
            
            with col2:
                is_grounded = not message.get("is_out_of_scope", False)
                status = "🟢 Grounded" if is_grounded else "🟡 Refusal"
                st.caption(status)
            
            with col3:
                timestamp = message.get("timestamp", "")
                if timestamp:
                    try:
                        dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                        st.caption(f"🕐 {dt.strftime('%H:%M:%S')}")
                    except:
                        st.caption(f"🕐 {timestamp}")


def render_sidebar():
    """Render sidebar with user info and conversations."""
    with st.sidebar:
        # User info
        st.markdown(f"### 👤 {st.session_state.user['email']}")
        if st.button("🚪 Logout", use_container_width=True):
            logout_user()
        
        st.markdown("---")
        
        # Organization/Namespace selector
        st.subheader("🏢 Knowledge Base")
        
        if st.button("🔄 Refresh", key="refresh_orgs"):
            fetch_organizations()
            fetch_websites()
            fetch_conversations()
        
        # Fetch on first load
        if not st.session_state.organizations:
            fetch_organizations()
            fetch_websites()
            fetch_conversations()
        
        # Create namespace options
        namespace_options = {"Zibtek (default)": "zibtek"}
        for website in st.session_state.websites:
            org = next((o for o in st.session_state.organizations if o['id'] == website['org_id']), None)
            org_name = org['name'] if org else "Unknown"
            label = f"{org_name} - {website['url']}"
            namespace_options[label] = website['namespace']
        
        selected_label = st.selectbox(
            "Select Organization:",
            list(namespace_options.keys()),
            index=list(namespace_options.values()).index(st.session_state.current_namespace) 
                if st.session_state.current_namespace in namespace_options.values() else 0,
            key="namespace_selector"
        )
        
        new_namespace = namespace_options[selected_label]
        if new_namespace != st.session_state.current_namespace:
            st.session_state.current_namespace = new_namespace
            st.session_state.current_session_id = str(uuid.uuid4())
            st.session_state.chat_history = []
        
        st.caption(f"📦 Namespace: `{st.session_state.current_namespace}`")
        
        st.markdown("---")
        
        # Conversations
        st.subheader("💬 Conversations")
        
        if st.button("➕ New Conversation", use_container_width=True):
            st.session_state.current_session_id = str(uuid.uuid4())
            st.session_state.chat_history = []
            st.rerun()
        
        # Show conversations for current namespace
        namespace_convs = st.session_state.conversations.get(st.session_state.current_namespace, [])
        
        if namespace_convs:
            for conv in namespace_convs[:10]:  # Show last 10
                col1, col2 = st.columns([3, 1])
                with col1:
                    title = conv.get("title", "Untitled")
                    if len(title) > 30:
                        title = title[:27] + "..."
                    
                    if st.button(f"💬 {title}", key=f"conv_{conv['id']}", use_container_width=True):
                        fetch_conversation_messages(conv['namespace'], conv['session_id'])
                        st.rerun()
                
                with col2:
                    if st.button("🗑️", key=f"del_{conv['id']}"):
                        delete_conversation(conv['namespace'], conv['session_id'])
                        st.rerun()
        else:
            st.caption("No conversations yet")


def render_chat_page():
    """Render main chat interface."""
    st.markdown("<h1 style='text-align: center;'>🤖 Multi-Tenant AI Assistant</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #666;'>Ask questions about any organization's website content</p>", unsafe_allow_html=True)
    
    render_sidebar()
    
    # Chat history
    chat_container = st.container()
    
    with chat_container:
        if not st.session_state.chat_history:
            st.info("👋 Hello! Ask me anything about the selected organization's content!")
        else:
            for message in st.session_state.chat_history:
                render_message(message, message.get("is_user", False))
    
    # Chat input
    st.markdown("---")
    
    with st.form(key="chat_form", clear_on_submit=True):
        user_input = st.text_input(
            "Your question:",
            placeholder="e.g., What services does this organization offer?",
            label_visibility="collapsed"
        )
        
        send_clicked = st.form_submit_button("📤 Send", type="primary", use_container_width=True)
    
    if send_clicked and user_input and user_input.strip():
        question = user_input.strip()
        
        # Add user message to history
        user_message = {
            "content": question,
            "is_user": True,
            "timestamp": datetime.now().isoformat()
        }
        st.session_state.chat_history.append(user_message)
        
        # Show user message immediately
        with chat_container:
            render_message(user_message, True)
        
        # Show "thinking" indicator
        with st.spinner("🤔 Thinking..."):
            response = send_message(question, st.session_state.current_namespace)
        
        if response:
            # Add bot response to history
            bot_message = {
                "content": response.get("answer", "Sorry, I couldn't generate a response."),
                "citations": response.get("citations", []),
                "retrieval_steps": response.get("retrieval_steps", {}),
                "is_user": False,
                "is_out_of_scope": response.get("is_out_of_scope", False),
                "processing_time_ms": response.get("processing_time_ms", 0),
                "timestamp": datetime.now().isoformat()
            }
            st.session_state.chat_history.append(bot_message)
            
            # Refresh conversations list
            fetch_conversations()
        
        st.rerun()


def main():
    """Main application entry point."""
    st.set_page_config(
        page_title="Multi-Tenant AI Assistant",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
    .main-header {
        text-align: center;
        color: #1f77b4;
        margin-bottom: 10px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Initialize session
    initialize_session()
    
    # Check if user is logged in
    if not st.session_state.user or not st.session_state.token:
        render_auth_page()
    else:
        render_chat_page()


if __name__ == "__main__":
    main()
