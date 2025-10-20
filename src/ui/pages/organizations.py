"""Organization management UI page for Streamlit."""

import streamlit as st
import requests
import time
from typing import List, Dict, Any, Optional
import os

# Configuration
API_BASE_URL = os.getenv("API_BASE_URL", "https://zibtek-chatbot-42bb2cdc74d2.herokuapp.com")


def get_auth_headers():
    """Get authentication headers."""
    if 'token' in st.session_state and st.session_state.token:
        return {"Authorization": f"Bearer {st.session_state.token}"}
    return {}


def render_organizations_page():
    """Render the organization management page."""
    st.title("🏢 Organization Management")
    st.write("Manage organizations, websites, and trigger ingestion.")
    
    # Check if user is logged in
    if 'user' not in st.session_state or not st.session_state.user:
        st.warning("⚠️ Please login first from the main page")
        return
    
    # Tabs for different functions
    tab1, tab2, tab3 = st.tabs(["Organizations", "Websites", "Ingestion Jobs"])
    
    with tab1:
        render_organizations_tab()
    
    with tab2:
        render_websites_tab()
    
    with tab3:
        render_jobs_tab()


def render_organizations_tab():
    """Render organizations list and creation."""
    st.header("Organizations")
    
    # Create new organization
    with st.expander("➕ Create New Organization"):
        with st.form("create_org_form"):
            org_name = st.text_input("Organization Name", placeholder="e.g., Acme Corp")
            org_description = st.text_area("Description (optional)", placeholder="Describe this organization")
            
            submitted = st.form_submit_button("Create Organization")
            if submitted and org_name:
                create_organization(org_name, org_description)
    
    st.write("---")
    
    # List existing organizations
    st.subheader("Existing Organizations")
    orgs = fetch_organizations()
    
    if not orgs:
        st.info("No organizations found. Create one to get started!")
    else:
        for org in orgs:
            with st.container():
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.markdown(f"### {org['name']}")
                    st.caption(f"**Slug:** `{org['slug']}` | **ID:** `{org['id']}`")
                    if org.get('description'):
                        st.write(org['description'])
                with col2:
                    st.metric("Created", org['created_at'][:10])
                
                st.write("---")


def render_websites_tab():
    """Render websites list and creation."""
    st.header("Websites")
    
    # Get organizations for dropdown
    orgs = fetch_organizations()
    
    if not orgs:
        st.warning("Please create an organization first!")
        return
    
    # Create new website
    with st.expander("➕ Add New Website"):
        with st.form("create_website_form"):
            org_options = {org['name']: org['id'] for org in orgs}
            selected_org_name = st.selectbox("Select Organization", list(org_options.keys()))
            
            website_url = st.text_input(
                "Website URL",
                placeholder="https://example.com",
                help="Full URL of the website to crawl"
            )
            
            submitted = st.form_submit_button("Add Website")
            if submitted and website_url and selected_org_name:
                org_id = org_options[selected_org_name]
                create_website(org_id, website_url)
    
    st.write("---")
    
    # Filter by organization
    st.subheader("Existing Websites")
    filter_org = st.selectbox(
        "Filter by Organization",
        ["All"] + [org['name'] for org in orgs],
        key="website_filter"
    )
    
    org_id_filter = None
    if filter_org != "All":
        org_id_filter = next(org['id'] for org in orgs if org['name'] == filter_org)
    
    websites = fetch_websites(org_id_filter)
    
    if not websites:
        st.info("No websites found.")
    else:
        for website in websites:
            with st.container():
                col1, col2, col3 = st.columns([2, 1, 1])
                
                with col1:
                    st.markdown(f"**{website['url']}**")
                    st.caption(f"Namespace: `{website['namespace']}`")
                
                with col2:
                    status = website['status']
                    status_color = {
                        'pending': '🟡',
                        'ingesting': '🔵',
                        'completed': '🟢',
                        'failed': '🔴'
                    }.get(status, '⚪')
                    st.write(f"{status_color} **{status.upper()}**")
                    
                    if website.get('pages_crawled'):
                        st.caption(f"📄 {website['pages_crawled']} pages")
                    if website.get('chunks_created'):
                        st.caption(f"📦 {website['chunks_created']} chunks")
                
                with col3:
                    if website['status'] in ['pending', 'completed', 'failed']:
                        if st.button("Start Ingestion", key=f"ingest_{website['id']}"):
                            start_ingestion(website['id'])
                    elif website['status'] == 'ingesting':
                        st.button("Ingesting...", key=f"ingesting_{website['id']}", disabled=True)
                
                st.write("---")


def render_jobs_tab():
    """Render ingestion jobs and their status."""
    st.header("Ingestion Jobs")
    
    # Auto-refresh toggle
    auto_refresh = st.checkbox("Auto-refresh every 5 seconds", value=False)
    
    # Get all websites to map job to website
    websites = fetch_websites()
    website_map = {w['id']: w for w in websites}
    
    st.subheader("Recent Jobs")
    
    # For simplicity, we'll show jobs for all websites
    # In production, you might want pagination
    all_jobs = []
    for website in websites:
        jobs = fetch_jobs_for_website(website['id'])
        all_jobs.extend(jobs)
    
    # Sort by created_at descending
    all_jobs.sort(key=lambda x: x['created_at'], reverse=True)
    
    if not all_jobs:
        st.info("No ingestion jobs found.")
    else:
        has_active_jobs = False
        for job in all_jobs[:20]:  # Show latest 20
            website = website_map.get(job['website_id'])
            website_url = website['url'] if website else 'Unknown'
            
            # Fetch runtime status for running/pending jobs
            status = job['status']
            progress = job.get('progress_percent', 0)
            pages = job.get('pages_crawled', 0)
            chunks = job.get('chunks_created', 0)
            error = job.get('error_message')
            
            if status in ['pending', 'running']:
                has_active_jobs = True
                runtime_status = fetch_job_runtime_status(job['id'])
                if runtime_status:
                    status = runtime_status.get('db_status', status)
                    progress = runtime_status.get('progress_percent', progress)
                    pages = runtime_status.get('pages_crawled', pages)
                    chunks = runtime_status.get('chunks_created', chunks)
                    error = runtime_status.get('error_message', error)
            
            with st.container():
                col1, col2, col3 = st.columns([2, 1, 1])
                
                with col1:
                    st.markdown(f"**{website_url}**")
                    st.caption(f"Job ID: `{job['id'][:8]}...`")
                
                with col2:
                    status_icon = {
                        'pending': '⏳',
                        'running': '🔄',
                        'completed': '✅',
                        'failed': '❌'
                    }.get(status, '❓')
                    st.write(f"{status_icon} **{status.upper()}**")
                    
                    if progress > 0:
                        st.progress(progress / 100)
                        st.caption(f"{progress}%")
                
                with col3:
                    if pages > 0:
                        st.metric("Pages", pages)
                    if chunks > 0:
                        st.metric("Chunks", chunks)
                
                if error:
                    st.error(f"Error: {error}")
                
                st.write("---")
        
        # Auto-refresh after displaying all jobs
        if auto_refresh and has_active_jobs:
            st.info("🔄 Auto-refresh enabled - page will refresh in 5 seconds")
            time.sleep(5)
            st.rerun()
        elif auto_refresh and not has_active_jobs:
            st.success("✅ All jobs completed - auto-refresh disabled")


# API helper functions
def fetch_organizations() -> List[Dict[str, Any]]:
    """Fetch all organizations from API."""
    try:
        response = requests.get(
            f"{API_BASE_URL}/organizations",
            headers=get_auth_headers(),
            timeout=10
        )
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Failed to fetch organizations: {response.status_code}")
            return []
    except Exception as e:
        st.error(f"Error fetching organizations: {e}")
        return []


def create_organization(name: str, description: Optional[str] = None):
    """Create a new organization."""
    try:
        payload = {"name": name}
        if description:
            payload["description"] = description
        
        response = requests.post(
            f"{API_BASE_URL}/organizations",
            json=payload,
            headers=get_auth_headers(),
            timeout=10
        )
        
        if response.status_code == 200:
            st.success(f"✅ Organization '{name}' created successfully!")
            time.sleep(1)
            st.rerun()
        else:
            st.error(f"Failed to create organization: {response.text}")
    except Exception as e:
        st.error(f"Error creating organization: {e}")


def fetch_websites(org_id: Optional[str] = None) -> List[Dict[str, Any]]:
    """Fetch websites, optionally filtered by organization."""
    try:
        params = {"org_id": org_id} if org_id else {}
        response = requests.get(
            f"{API_BASE_URL}/websites",
            params=params,
            headers=get_auth_headers(),
            timeout=10
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Failed to fetch websites: {response.status_code}")
            return []
    except Exception as e:
        st.error(f"Error fetching websites: {e}")
        return []


def create_website(org_id: str, url: str):
    """Create a new website."""
    try:
        payload = {
            "org_id": org_id,
            "url": url
        }
        
        response = requests.post(
            f"{API_BASE_URL}/organizations/{org_id}/websites",
            json=payload,
            headers=get_auth_headers(),
            timeout=10
        )
        
        if response.status_code == 200:
            st.success(f"✅ Website '{url}' added successfully!")
            time.sleep(1)
            st.rerun()
        else:
            st.error(f"Failed to add website: {response.text}")
    except Exception as e:
        st.error(f"Error adding website: {e}")


def start_ingestion(website_id: str, max_pages: int = 500):
    """Start ingestion for a website."""
    try:
        response = requests.post(
            f"{API_BASE_URL}/websites/{website_id}/ingest",
            params={"max_pages": max_pages},
            headers=get_auth_headers(),
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            st.success(f"✅ Ingestion started! Job ID: {result['job_id']}")
            time.sleep(1)
            st.rerun()
        else:
            st.error(f"Failed to start ingestion: {response.text}")
    except Exception as e:
        st.error(f"Error starting ingestion: {e}")


def fetch_jobs_for_website(website_id: str) -> List[Dict[str, Any]]:
    """Fetch ingestion jobs for a website."""
    try:
        response = requests.get(
            f"{API_BASE_URL}/websites/{website_id}/jobs",
            headers=get_auth_headers(),
            timeout=10
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            return []
    except Exception:
        return []


def fetch_job_runtime_status(job_id: str) -> Optional[Dict[str, Any]]:
    """Fetch real-time job status from job manager."""
    try:
        response = requests.get(
            f"{API_BASE_URL}/jobs/{job_id}/status",
            headers=get_auth_headers(),
            timeout=10
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            return None
    except Exception:
        return None


if __name__ == "__main__":
    render_organizations_page()
