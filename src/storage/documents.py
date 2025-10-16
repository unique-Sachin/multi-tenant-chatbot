"""Database operations for document tracking."""

from typing import List, Dict, Any, Optional
from datetime import datetime
from pydantic import BaseModel
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from src.storage.db import supabase


class DocumentRecord(BaseModel):
    """Model for a document record."""
    id: str
    org_id: str
    user_id: str
    filename: str
    file_type: str
    file_size_bytes: int
    doc_id: str
    namespace: str
    chunk_count: int
    character_count: int
    status: str
    error_message: Optional[str] = None
    uploaded_at: datetime
    processed_at: Optional[datetime] = None
    metadata: Optional[Dict[str, Any]] = None


def create_document_record(
    org_id: str,
    user_id: str,
    filename: str,
    file_type: str,
    file_size_bytes: int,
    doc_id: str,
    namespace: str,
    metadata: Optional[Dict[str, Any]] = None
) -> Optional[str]:
    """Create a new document record in pending status.
    
    Returns:
        Document ID if successful, None otherwise
    """
    if not supabase:
        print("❌ Supabase client not initialized")
        return None
    
    try:
        doc_data = {
            'org_id': org_id,
            'user_id': user_id,
            'filename': filename,
            'file_type': file_type,
            'file_size_bytes': file_size_bytes,
            'doc_id': doc_id,
            'namespace': namespace,
            'status': 'processing',
            'metadata': metadata or {}
        }
        
        result = supabase.table("documents").insert(doc_data).execute()
        
        if result.data:
            print(f"✅ Created document record: {doc_id}")
            return result.data[0]['id']
        return None
        
    except Exception as e:
        print(f"❌ Error creating document record: {e}")
        return None


def update_document_success(
    doc_id: str,
    chunk_count: int,
    character_count: int
) -> bool:
    """Update document record after successful ingestion."""
    if not supabase:
        print("❌ Supabase client not initialized")
        return False
    
    try:
        update_data = {
            'status': 'completed',
            'chunk_count': chunk_count,
            'character_count': character_count,
            'processed_at': datetime.now().isoformat()
        }
        
        result = supabase.table("documents")\
            .update(update_data)\
            .eq("doc_id", doc_id)\
            .execute()
        
        if result.data:
            print(f"✅ Updated document record: {doc_id} -> completed")
            return True
        return False
        
    except Exception as e:
        print(f"❌ Error updating document record: {e}")
        return False


def update_document_failure(
    doc_id: str,
    error_message: str
) -> bool:
    """Update document record after failed ingestion."""
    if not supabase:
        print("❌ Supabase client not initialized")
        return False
    
    try:
        update_data = {
            'status': 'failed',
            'error_message': error_message,
            'processed_at': datetime.now().isoformat()
        }
        
        result = supabase.table("documents")\
            .update(update_data)\
            .eq("doc_id", doc_id)\
            .execute()
        
        if result.data:
            print(f"✅ Updated document record: {doc_id} -> failed")
            return True
        return False
        
    except Exception as e:
        print(f"❌ Error updating document failure: {e}")
        return False


def get_documents_by_org(org_id: str) -> List[DocumentRecord]:
    """Get all documents for an organization."""
    if not supabase:
        print("❌ Supabase client not initialized")
        return []
    
    try:
        result = supabase.table("documents")\
            .select("*")\
            .eq("org_id", org_id)\
            .order("uploaded_at", desc=True)\
            .execute()
        
        if result.data:
            return [DocumentRecord(**doc) for doc in result.data]
        return []
        
    except Exception as e:
        print(f"❌ Error getting documents by org: {e}")
        return []


def get_documents_by_namespace(namespace: str) -> List[DocumentRecord]:
    """Get all documents in a namespace."""
    if not supabase:
        print("❌ Supabase client not initialized")
        return []
    
    try:
        result = supabase.table("documents")\
            .select("*")\
            .eq("namespace", namespace)\
            .order("uploaded_at", desc=True)\
            .execute()
        
        if result.data:
            return [DocumentRecord(**doc) for doc in result.data]
        return []
        
    except Exception as e:
        print(f"❌ Error getting documents by namespace: {e}")
        return []


def get_document_by_doc_id(doc_id: str) -> Optional[DocumentRecord]:
    """Get a single document by its doc_id."""
    if not supabase:
        print("❌ Supabase client not initialized")
        return None
    
    try:
        result = supabase.table("documents")\
            .select("*")\
            .eq("doc_id", doc_id)\
            .execute()
        
        if result.data:
            return DocumentRecord(**result.data[0])
        return None
        
    except Exception as e:
        print(f"❌ Error getting document by doc_id: {e}")
        return None


def delete_document_record(doc_id: str) -> bool:
    """Delete a document record."""
    if not supabase:
        print("❌ Supabase client not initialized")
        return False
    
    try:
        result = supabase.table("documents")\
            .delete()\
            .eq("doc_id", doc_id)\
            .execute()
        
        if result.data:
            print(f"✅ Deleted document record: {doc_id}")
            return True
        return False
        
    except Exception as e:
        print(f"❌ Error deleting document: {e}")
        return False


def get_document_stats(org_id: Optional[str] = None) -> Dict[str, Any]:
    """Get statistics about documents."""
    if not supabase:
        print("❌ Supabase client not initialized")
        return {}
    
    try:
        query = supabase.table("documents").select("status, file_type")
        
        if org_id:
            query = query.eq("org_id", org_id)
        
        result = query.execute()
        
        if result.data:
            stats = {
                'total': len(result.data),
                'by_status': {},
                'by_type': {}
            }
            
            for doc in result.data:
                status = doc.get('status', 'unknown')
                file_type = doc.get('file_type', 'unknown')
                
                stats['by_status'][status] = stats['by_status'].get(status, 0) + 1
                stats['by_type'][file_type] = stats['by_type'].get(file_type, 0) + 1
            
            return stats
        
        return {'total': 0, 'by_status': {}, 'by_type': {}}
        
    except Exception as e:
        print(f"❌ Error getting document stats: {e}")
        return {}
