"""Ingestion runner with progress tracking for multi-tenant system."""

import sys
import os
from typing import Optional, Dict, Any

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.ingest.ingest import ZibtekIngestor
from src.storage.organizations import (
    get_website,
    update_website_status,
    update_ingestion_job
)


def run_ingestion_job(
    website_id: str,
    job_id: str,
    domain: str,
    namespace: str,
    max_pages: int = 50
) -> Dict[str, Any]:
    """Run ingestion pipeline for a website with progress tracking.
    
    Args:
        website_id: Website ID from database
        job_id: Ingestion job ID
        domain: Website URL to crawl
        namespace: Pinecone namespace for this organization
        max_pages: Maximum pages to crawl
        
    Returns:
        Dict with result information
    """
    try:
        # Update job status to running
        update_ingestion_job(job_id, status="running", progress_percent=0)
        update_website_status(website_id, status="ingesting")
        
        # Initialize ingestor
        print(f"🚀 Starting ingestion for {domain} (namespace: {namespace})")
        ingestor = ZibtekIngestor(
            domain=domain,
            namespace=namespace,
            max_pages=max_pages
        )
        
        # Step 1: Discover URLs (10%)
        update_ingestion_job(job_id, progress_percent=10)
        urls = ingestor.discover_urls()
        
        if not urls:
            raise Exception("No URLs found to crawl")
        
        # Step 2: Crawl pages (40%)
        update_ingestion_job(job_id, progress_percent=20)
        pages = ingestor.crawl_pages(urls)
        
        if not pages:
            raise Exception("No content extracted from pages")
        
        update_ingestion_job(
            job_id,
            progress_percent=40,
            pages_crawled=len(pages)
        )
        
        # Step 3: Chunk content (60%)
        chunks = ingestor.chunk_content(pages)
        
        if not chunks:
            raise Exception("No chunks created from content")
        
        update_ingestion_job(
            job_id,
            progress_percent=60,
            chunks_created=len(chunks)
        )
        
        # Step 4: Generate embeddings (80%)
        chunks_with_embeddings = ingestor.generate_embeddings(chunks)
        update_ingestion_job(job_id, progress_percent=80)
        
        # Step 5: Store in Pinecone (100%)
        success = ingestor.store_in_pinecone(chunks_with_embeddings)
        
        if not success:
            raise Exception("Failed to store chunks in Pinecone")
        
        # Update final status
        update_ingestion_job(
            job_id,
            status="completed",
            progress_percent=100,
            pages_crawled=len(pages),
            chunks_created=len(chunks)
        )
        
        update_website_status(
            website_id,
            status="completed",
            pages_crawled=len(pages),
            chunks_created=len(chunks)
        )
        
        result = {
            "success": True,
            "pages_crawled": len(pages),
            "chunks_created": len(chunks),
            "namespace": namespace
        }
        
        print(f"✅ Ingestion completed for {domain}")
        return result
        
    except Exception as e:
        error_msg = str(e)
        print(f"❌ Ingestion failed: {error_msg}")
        
        # Update job and website with error
        update_ingestion_job(
            job_id,
            status="failed",
            error_message=error_msg
        )
        
        update_website_status(website_id, status="failed")
        
        return {
            "success": False,
            "error": error_msg
        }
