"""Organization and website management for multi-tenant ingestion."""

import os
import re
import uuid
from datetime import datetime
from typing import Optional, Dict, Any, List
from supabase import Client
from pydantic import BaseModel
from dotenv import load_dotenv

from .db import supabase

load_dotenv()


# Pydantic models
class OrganizationCreate(BaseModel):
    """Model for creating an organization."""
    name: str
    description: Optional[str] = None


class OrganizationResponse(BaseModel):
    """Model for organization response."""
    id: str
    name: str
    slug: str
    description: Optional[str]
    created_at: datetime
    updated_at: datetime


class WebsiteCreate(BaseModel):
    """Model for creating a website."""
    org_id: str
    url: str


class WebsiteResponse(BaseModel):
    """Model for website response."""
    id: str
    org_id: str
    url: str
    namespace: str
    status: str
    pages_crawled: int
    chunks_created: int
    last_ingested_at: Optional[datetime]
    created_at: datetime
    updated_at: datetime


class IngestionJobCreate(BaseModel):
    """Model for creating an ingestion job."""
    website_id: str


class IngestionJobResponse(BaseModel):
    """Model for ingestion job response."""
    id: str
    website_id: str
    status: str
    progress_percent: int
    pages_crawled: int
    chunks_created: int
    error_message: Optional[str]
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    created_at: datetime


# Helper functions
def create_slug(name: str) -> str:
    """Create URL-safe slug from organization name."""
    slug = name.lower()
    slug = re.sub(r'[^a-z0-9]+', '-', slug)
    slug = slug.strip('-')
    return slug


def create_namespace(org_slug: str) -> str:
    """Create namespace for organization.
    
    All websites and documents under the same organization share one namespace.
    This enables cross-website/document search within the organization.
    """
    # Use only org slug as namespace (ignore website_url for simplicity)
    namespace = org_slug.lower()
    namespace = re.sub(r'[^a-z0-9]+', '-', namespace)
    return namespace[:63]  # Pinecone namespace limit


# Organization CRUD
def create_organization(org_data: OrganizationCreate) -> Optional[OrganizationResponse]:
    """Create a new organization."""
    if not supabase:
        print("❌ Supabase client not initialized")
        return None
    
    slug = create_slug(org_data.name)
    
    data = {
        "name": org_data.name,
        "slug": slug,
        "description": org_data.description
    }
    
    try:
        result = supabase.table("organizations").insert(data).execute()
        if hasattr(result, 'data') and result.data:
            return OrganizationResponse(**result.data[0])
        return None
    except Exception as e:
        print(f"Error creating organization: {e}")
        return None


def get_organization(org_id: str) -> Optional[OrganizationResponse]:
    """Get organization by ID."""
    if not supabase:
        return None
    
    try:
        result = supabase.table("organizations").select("*").eq("id", org_id).execute()
        if hasattr(result, 'data') and result.data:
            return OrganizationResponse(**result.data[0])
        return None
    except Exception as e:
        print(f"Error fetching organization: {e}")
        return None


def get_organization_by_slug(slug: str) -> Optional[OrganizationResponse]:
    """Get organization by slug."""
    if not supabase:
        return None
    
    try:
        result = supabase.table("organizations").select("*").eq("slug", slug).execute()
        if hasattr(result, 'data') and result.data:
            return OrganizationResponse(**result.data[0])
        return None
    except Exception as e:
        print(f"Error fetching organization by slug: {e}")
        return None


def get_organization_by_namespace(namespace: str) -> Optional[Dict[str, Any]]:
    """Get organization and website info by namespace.
    
    Args:
        namespace: The namespace to look up
        
    Returns:
        Dict with organization and website info, or None if not found
    """
    if not supabase:
        return None
    
    # Handle default Zibtek namespace
    if namespace == "zibtek":
        return {
            "org_name": "Zibtek",
            "org_description": "Leading software development company",
            "website_url": "https://zibtek.com",
            "namespace": "zibtek"
        }
    
    try:
        # Get website by namespace first
        website_result = supabase.table("websites").select("*, organizations(*)").eq("namespace", namespace).execute()
        
        if hasattr(website_result, 'data') and website_result.data:
            website = website_result.data[0]
            org = website.get('organizations')
            
            if org:
                return {
                    "org_name": org.get('name', 'Unknown Organization'),
                    "org_description": org.get('description', ''),
                    "website_url": website.get('url', ''),
                    "namespace": namespace
                }
        
        return None
    except Exception as e:
        print(f"Error fetching organization by namespace: {e}")
        return None


def list_organizations() -> List[OrganizationResponse]:
    """List all organizations."""
    if not supabase:
        return []
    
    try:
        result = supabase.table("organizations").select("*").order("created_at", desc=True).execute()
        if hasattr(result, 'data') and result.data:
            return [OrganizationResponse(**org) for org in result.data]
        return []
    except Exception as e:
        print(f"Error listing organizations: {e}")
        return []


# Website CRUD
def create_website(website_data: WebsiteCreate) -> Optional[WebsiteResponse]:
    """Create a new website for an organization."""
    if not supabase:
        print("❌ Supabase client not initialized")
        return None
    
    # Get organization to create namespace
    org = get_organization(website_data.org_id)
    if not org:
        print(f"❌ Organization {website_data.org_id} not found")
        return None
    
    namespace = create_namespace(org.slug)
    
    data = {
        "org_id": website_data.org_id,
        "url": website_data.url,
        "namespace": namespace,
        "status": "pending"
    }
    
    try:
        result = supabase.table("websites").insert(data).execute()
        if hasattr(result, 'data') and result.data:
            return WebsiteResponse(**result.data[0])
        return None
    except Exception as e:
        print(f"Error creating website: {e}")
        return None


def get_website(website_id: str) -> Optional[WebsiteResponse]:
    """Get website by ID."""
    if not supabase:
        return None
    
    try:
        result = supabase.table("websites").select("*").eq("id", website_id).execute()
        if hasattr(result, 'data') and result.data:
            return WebsiteResponse(**result.data[0])
        return None
    except Exception as e:
        print(f"Error fetching website: {e}")
        return None


def list_websites(org_id: Optional[str] = None) -> List[WebsiteResponse]:
    """List websites, optionally filtered by organization."""
    if not supabase:
        return []
    
    try:
        query = supabase.table("websites").select("*")
        if org_id:
            query = query.eq("org_id", org_id)
        result = query.order("created_at", desc=True).execute()
        
        if hasattr(result, 'data') and result.data:
            return [WebsiteResponse(**website) for website in result.data]
        return []
    except Exception as e:
        print(f"Error listing websites: {e}")
        return []


def update_website_status(
    website_id: str, 
    status: str,
    pages_crawled: Optional[int] = None,
    chunks_created: Optional[int] = None
) -> bool:
    """Update website status and metrics."""
    if not supabase:
        return False
    
    data = {
        "status": status,
        "updated_at": datetime.now().isoformat()
    }
    
    if pages_crawled is not None:
        data["pages_crawled"] = pages_crawled
    
    if chunks_created is not None:
        data["chunks_created"] = chunks_created
    
    if status == "completed":
        data["last_ingested_at"] = datetime.now().isoformat()
    
    try:
        supabase.table("websites").update(data).eq("id", website_id).execute()
        return True
    except Exception as e:
        print(f"Error updating website status: {e}")
        return False


# Ingestion job CRUD
def create_ingestion_job(job_data: IngestionJobCreate) -> Optional[IngestionJobResponse]:
    """Create a new ingestion job."""
    if not supabase:
        print("❌ Supabase client not initialized")
        return None
    
    data = {
        "website_id": job_data.website_id,
        "status": "pending"
    }
    
    try:
        result = supabase.table("ingestion_jobs").insert(data).execute()
        if hasattr(result, 'data') and result.data:
            return IngestionJobResponse(**result.data[0])
        return None
    except Exception as e:
        print(f"Error creating ingestion job: {e}")
        return None


def get_ingestion_job(job_id: str) -> Optional[IngestionJobResponse]:
    """Get ingestion job by ID."""
    if not supabase:
        return None
    
    try:
        result = supabase.table("ingestion_jobs").select("*").eq("id", job_id).execute()
        if hasattr(result, 'data') and result.data:
            return IngestionJobResponse(**result.data[0])
        return None
    except Exception as e:
        print(f"Error fetching ingestion job: {e}")
        return None


def list_ingestion_jobs(website_id: Optional[str] = None) -> List[IngestionJobResponse]:
    """List ingestion jobs, optionally filtered by website."""
    if not supabase:
        return []
    
    try:
        query = supabase.table("ingestion_jobs").select("*")
        if website_id:
            query = query.eq("website_id", website_id)
        result = query.order("created_at", desc=True).execute()
        
        if hasattr(result, 'data') and result.data:
            return [IngestionJobResponse(**job) for job in result.data]
        return []
    except Exception as e:
        print(f"Error listing ingestion jobs: {e}")
        return []


def update_ingestion_job(
    job_id: str,
    status: Optional[str] = None,
    progress_percent: Optional[int] = None,
    pages_crawled: Optional[int] = None,
    chunks_created: Optional[int] = None,
    error_message: Optional[str] = None
) -> bool:
    """Update ingestion job progress and status."""
    if not supabase:
        return False
    
    data = {}
    
    if status:
        data["status"] = status
        if status == "running" and not data.get("started_at"):
            data["started_at"] = datetime.now().isoformat()
        elif status in ["completed", "failed"]:
            data["completed_at"] = datetime.now().isoformat()
    
    if progress_percent is not None:
        data["progress_percent"] = progress_percent
    
    if pages_crawled is not None:
        data["pages_crawled"] = pages_crawled
    
    if chunks_created is not None:
        data["chunks_created"] = chunks_created
    
    if error_message is not None:
        data["error_message"] = error_message
    
    try:
        supabase.table("ingestion_jobs").update(data).eq("id", job_id).execute()
        return True
    except Exception as e:
        print(f"Error updating ingestion job: {e}")
        return False
