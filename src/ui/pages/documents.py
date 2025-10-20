"""Document upload page for ingesting PDF, TXT, DOCX files."""

import streamlit as st
import requests
import os
from pathlib import Path

# Configuration
API_BASE_URL = os.getenv("API_BASE_URL", "https://zibtek-chatbot-747d3d71730d.herokuapp.com")


def get_auth_headers():
    """Get authentication headers."""
    if 'token' in st.session_state and st.session_state.token:
        return {"Authorization": f"Bearer {st.session_state.token}"}
    return {}


def fetch_organizations():
    """Fetch organizations list."""
    try:
        response = requests.get(
            f"{API_BASE_URL}/organizations",
            headers=get_auth_headers(),
            timeout=30
        )
        if response.status_code == 200:
            return response.json()
        return []
    except Exception as e:
        st.error(f"Error fetching organizations: {e}")
        return []


def fetch_websites():
    """Fetch all websites."""
    try:
        response = requests.get(
            f"{API_BASE_URL}/websites",
            headers=get_auth_headers(),
            timeout=30
        )
        if response.status_code == 200:
            return response.json()
        return []
    except Exception as e:
        st.error(f"Error fetching websites: {e}")
        return []


def fetch_documents_by_org(org_id):
    """Fetch documents for an organization."""
    try:
        response = requests.get(
            f"{API_BASE_URL}/documents/org/{org_id}",
            headers=get_auth_headers(),
            timeout=30
        )
        if response.status_code == 200:
            return response.json().get('documents', [])
        return []
    except Exception as e:
        st.error(f"Error fetching documents: {e}")
        return []



def upload_document(file, org_id):
    """Upload document to backend."""
    try:
        files = {"file": (file.name, file.getvalue(), file.type)}
        data = {"org_id": org_id}
        
        response = requests.post(
            f"{API_BASE_URL}/documents/upload",
            files=files,
            data=data,
            headers=get_auth_headers(),
            timeout=120  # 2 minutes for large files
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            error_detail = response.json().get('detail', response.text)
            st.error(f"Upload failed: {error_detail}")
            return None
            
    except Exception as e:
        st.error(f"Error uploading document: {e}")
        return None


def main():
    """Document upload page."""
    st.title("📄 Document Upload")
    st.markdown("Upload PDF, TXT, or Word documents to add them to your knowledge base")
    
    # Check if user is logged in
    if 'user' not in st.session_state or not st.session_state.user:
        st.warning("⚠️ Please login first from the main page")
        return
    
    st.markdown("---")
    
    # Fetch organizations and websites
    organizations = fetch_organizations()
    websites = fetch_websites()
    
    if not organizations:
        st.info("ℹ️ No organizations found. Create an organization first in the Organizations page.")
        return
    
    # Create organization selector
    org_options = {org['name']: org for org in organizations}
    
    selected_org_name = st.selectbox(
        "Select Organization",
        options=list(org_options.keys()),
        help="Choose which organization to upload the document to"
    )
    
    selected_org = org_options[selected_org_name]
    
    # Namespace is based on org slug
    namespace = selected_org['slug']
    
    st.info(f"📦 Namespace: `{namespace}` (all content for {selected_org['name']} organization)")
    
    # Show which websites are in this organization (if any)
    org_websites = [ws for ws in websites if ws['org_id'] == selected_org['id']]
    if org_websites:
        with st.expander(f"🌐 {len(org_websites)} website(s) in this organization"):
            for ws in org_websites:
                st.caption(f"• {ws['url']}")
    else:
        st.caption("💡 No websites ingested yet for this organization")
    
    # File uploader
    st.markdown("### Upload Document")
    
    uploaded_file = st.file_uploader(
        "Choose a file",
        type=['pdf', 'txt', 'docx'],
        help="Supported formats: PDF, TXT, DOCX (max 10MB)",
        accept_multiple_files=False
    )
    
    if uploaded_file:
        # Show file info
        file_size_mb = len(uploaded_file.getvalue()) / (1024 * 1024)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Filename", uploaded_file.name)
        with col2:
            st.metric("Size", f"{file_size_mb:.2f} MB")
        with col3:
            st.metric("Type", Path(uploaded_file.name).suffix.upper())
        
        # Validate file size
        if file_size_mb > 10:
            st.error("❌ File too large! Maximum size is 10MB")
            return
        
        st.markdown("---")
        
        # Upload button
        if st.button("🚀 Upload & Ingest", type="primary", use_container_width=True):
            with st.spinner(f"Uploading and processing {uploaded_file.name}..."):
                result = upload_document(
                    file=uploaded_file,
                    org_id=selected_org['id']
                )
                
                if result and result.get('success'):
                    st.success("✅ Document uploaded successfully!")
                    
                    # Show results
                    st.markdown("### Ingestion Results")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Chunks Created", result['chunks'])
                    with col2:
                        st.metric("Characters", f"{result['characters']:,}")
                    with col3:
                        st.metric("Time", f"{result['elapsed_seconds']}s")
                    with col4:
                        st.metric("Namespace", result['namespace'])
                    
                    st.info(f"🆔 Document ID: `{result['doc_id']}`")
                    
                    st.markdown("---")
                    st.success("🎉 Your document is now searchable in the knowledge base!")
                    
                    # Option to upload another
                    if st.button("📤 Upload Another Document"):
                        st.rerun()
    
    # Show existing documents
    st.markdown("---")
    st.markdown("### 📚 Uploaded Documents")
    
    documents = fetch_documents_by_org(selected_org['id'])
    
    if documents:
        for doc in documents:
            with st.expander(f"📄 {doc['filename']} ({doc['file_type']})"):
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    status_emoji = "✅" if doc['status'] == 'completed' else ("⏳" if doc['status'] == 'processing' else "❌")
                    st.metric("Status", f"{status_emoji} {doc['status'].title()}")
                with col2:
                    st.metric("Chunks", doc.get('chunk_count', 0))
                with col3:
                    file_size_mb = doc['file_size_bytes'] / (1024 * 1024)
                    st.metric("Size", f"{file_size_mb:.2f} MB")
                with col4:
                    uploaded_date = doc['uploaded_at'].split('T')[0]
                    st.metric("Uploaded", uploaded_date)
                
                st.caption(f"**Doc ID:** `{doc['doc_id']}`")
                st.caption(f"**Namespace:** `{doc['namespace']}`")
                
                if doc.get('error_message'):
                    st.error(f"❌ Error: {doc['error_message']}")
                
    else:
        st.info("No documents uploaded yet for this organization")
    
    # Info section
    st.markdown("---")
    st.markdown("### ℹ️ Information")
    
    with st.expander("📋 Supported Formats"):
        st.markdown("""
        - **PDF** (.pdf) - Portable Document Format
        - **Text** (.txt) - Plain text files
        - **Word** (.docx) - Microsoft Word documents
        
        ⚠️ Legacy .doc format is not supported. Please convert to .docx first.
        """)
    
    with st.expander("📏 File Size Limits"):
        st.markdown("""
        - Maximum file size: **10 MB**
        - For larger documents, consider splitting them into smaller files
        - Text files are typically very small (< 1MB)
        - PDF files with many images may be large
        """)
    
    with st.expander("🔍 How It Works"):
        st.markdown("""
        1. **Upload**: Your document is securely uploaded to the server
        2. **Extract**: Text content is extracted from the document
        3. **Chunk**: Text is split into manageable chunks (512 tokens)
        4. **Embed**: Each chunk is converted to a vector embedding
        5. **Store**: Embeddings are stored in Pinecone with metadata
        6. **Search**: Your document becomes instantly searchable via the chatbot!
        
        Documents are stored in the same namespace as website content for the selected organization.
        """)
    
    with st.expander("🔐 Privacy & Security"):
        st.markdown("""
        - Documents are associated with your user account
        - Only accessible within your organization's namespace
        - Metadata includes: filename, uploader, upload time
        - Original files are not stored, only text content and embeddings
        """)


if __name__ == "__main__":
    main()
