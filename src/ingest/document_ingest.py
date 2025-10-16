"""Document ingestion for PDF, TXT, DOCX files."""

import os
import hashlib
from datetime import datetime
from typing import List, Dict, Any, Optional, BinaryIO
from pathlib import Path
import tempfile

import tiktoken
from openai import OpenAI
from dotenv import load_dotenv

# Document parsing libraries
try:
    import PyPDF2
except ImportError:
    PyPDF2 = None

try:
    from docx import Document as DocxDocument
except ImportError:
    DocxDocument = None

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.storage.pine import init_index, upsert_chunks
from src.storage.documents import (
    create_document_record, 
    update_document_success, 
    update_document_failure
)

load_dotenv()


class DocumentIngestor:
    """Document ingestion pipeline for PDF, TXT, DOCX files."""
    
    SUPPORTED_FORMATS = {
        '.pdf': 'PDF Document',
        '.txt': 'Text File',
        '.docx': 'Word Document',
        '.doc': 'Word Document (Legacy)',
    }
    
    def __init__(self, namespace: str = "zibtek"):
        """Initialize document ingestor.
        
        Args:
            namespace: Pinecone namespace for this organization
        """
        self.namespace = namespace
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        
        self.client = OpenAI(api_key=self.openai_api_key)
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        self.chunk_size = 512
        self.chunk_overlap = 50
        
        # Initialize Pinecone
        self.index = init_index()
        
        print(f"📄 Initialized DocumentIngestor for namespace: {self.namespace}")
    
    def extract_text_from_pdf(self, file_content: bytes, filename: str) -> str:
        """Extract text from PDF file."""
        if not PyPDF2:
            raise ImportError("PyPDF2 is required for PDF processing. Install with: pip install PyPDF2")
        
        try:
            # Write to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                tmp_file.write(file_content)
                tmp_path = tmp_file.name
            
            # Extract text
            text_content = []
            with open(tmp_path, 'rb') as pdf_file:
                pdf_reader = PyPDF2.PdfReader(pdf_file)
                num_pages = len(pdf_reader.pages)
                
                print(f"  📖 Processing {num_pages} pages from {filename}")
                
                for page_num, page in enumerate(pdf_reader.pages, 1):
                    text = page.extract_text()
                    if text.strip():
                        text_content.append(text)
                        print(f"    ✓ Page {page_num}/{num_pages}")
            
            # Cleanup
            os.unlink(tmp_path)
            
            full_text = "\n\n".join(text_content)
            print(f"  ✅ Extracted {len(full_text)} characters from PDF")
            return full_text
            
        except Exception as e:
            print(f"  ❌ Error extracting PDF text: {e}")
            if 'tmp_path' in locals():
                os.unlink(tmp_path)
            raise
    
    def extract_text_from_docx(self, file_content: bytes, filename: str) -> str:
        """Extract text from DOCX file."""
        if not DocxDocument:
            raise ImportError("python-docx is required for DOCX processing. Install with: pip install python-docx")
        
        try:
            # Write to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.docx') as tmp_file:
                tmp_file.write(file_content)
                tmp_path = tmp_file.name
            
            # Extract text
            doc = DocxDocument(tmp_path)
            paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
            
            print(f"  📄 Extracted {len(paragraphs)} paragraphs from {filename}")
            
            # Cleanup
            os.unlink(tmp_path)
            
            full_text = "\n\n".join(paragraphs)
            print(f"  ✅ Extracted {len(full_text)} characters from DOCX")
            return full_text
            
        except Exception as e:
            print(f"  ❌ Error extracting DOCX text: {e}")
            if 'tmp_path' in locals():
                os.unlink(tmp_path)
            raise
    
    def extract_text_from_txt(self, file_content: bytes, filename: str) -> str:
        """Extract text from TXT file."""
        try:
            # Try UTF-8 first, fallback to latin-1
            try:
                text = file_content.decode('utf-8')
            except UnicodeDecodeError:
                text = file_content.decode('latin-1')
            
            print(f"  ✅ Extracted {len(text)} characters from TXT")
            return text
            
        except Exception as e:
            print(f"  ❌ Error extracting TXT text: {e}")
            raise
    
    def extract_text(self, file_content: bytes, filename: str) -> str:
        """Extract text from uploaded file based on extension."""
        file_ext = Path(filename).suffix.lower()
        
        if file_ext == '.pdf':
            return self.extract_text_from_pdf(file_content, filename)
        elif file_ext == '.docx':
            return self.extract_text_from_docx(file_content, filename)
        elif file_ext == '.txt':
            return self.extract_text_from_txt(file_content, filename)
        elif file_ext == '.doc':
            raise ValueError("Legacy .doc format not supported. Please convert to .docx")
        else:
            raise ValueError(f"Unsupported file format: {file_ext}")
    
    def chunk_text(self, text: str, source: str) -> List[Dict[str, Any]]:
        """Split text into overlapping chunks."""
        tokens = self.tokenizer.encode(text)
        chunks = []
        
        for i in range(0, len(tokens), self.chunk_size - self.chunk_overlap):
            chunk_tokens = tokens[i:i + self.chunk_size]
            chunk_text = self.tokenizer.decode(chunk_tokens)
            
            if chunk_text.strip():
                chunks.append({
                    'text': chunk_text,
                    'source': source,
                    'chunk_index': len(chunks),
                    'token_count': len(chunk_tokens)
                })
        
        return chunks
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for text chunks."""
        response = self.client.embeddings.create(
            model="text-embedding-3-small",
            input=texts
        )
        return [item.embedding for item in response.data]
    
    def ingest_document(
        self, 
        file_content: bytes, 
        filename: str,
        org_id: Optional[str] = None,
        user_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Ingest a single document file.
        
        Args:
            file_content: Raw file bytes
            filename: Original filename
            org_id: Organization ID (required for tracking)
            user_id: User ID who uploaded (required for tracking)
            metadata: Additional metadata to store with chunks
            
        Returns:
            Dict with ingestion results
        """
        start_time = datetime.now()
        print(f"\n📄 Ingesting document: {filename}")
        print(f"🏢 Namespace: {self.namespace}")
        
        # Generate document ID early for tracking
        doc_id = hashlib.sha256(file_content).hexdigest()[:16]
        file_size = len(file_content)
        
        try:
            # Validate file format
            file_ext = Path(filename).suffix.lower()
            if file_ext not in self.SUPPORTED_FORMATS:
                raise ValueError(f"Unsupported file format: {file_ext}")
            
            # Create document record in database (status: processing)
            if org_id and user_id:
                db_record_id = create_document_record(
                    org_id=org_id,
                    user_id=user_id,
                    filename=filename,
                    file_type=self.SUPPORTED_FORMATS[file_ext],
                    file_size_bytes=file_size,
                    doc_id=doc_id,
                    namespace=self.namespace,
                    metadata=metadata
                )
                print(f"  📝 Created database record: {db_record_id}")
            
            # Extract text
            print(f"  📖 Extracting text from {self.SUPPORTED_FORMATS[file_ext]}...")
            text_content = self.extract_text(file_content, filename)
            
            if not text_content or len(text_content.strip()) < 100:
                error_msg = "Insufficient text content extracted (minimum 100 characters)"
                if org_id and user_id:
                    update_document_failure(doc_id, error_msg)
                raise ValueError(error_msg)
            
            # Chunk text
            print(f"  ✂️  Chunking text...")
            chunks = self.chunk_text(text_content, f"document:{filename}")
            print(f"  ✅ Created {len(chunks)} chunks")
            
            # Generate embeddings
            print(f"  🧮 Generating embeddings...")
            chunk_texts = [chunk['text'] for chunk in chunks]
            embeddings = self.generate_embeddings(chunk_texts)
            
            # Prepare vectors for Pinecone
            vectors = []
            base_metadata = {
                'source_type': 'document',
                'filename': filename,
                'file_type': self.SUPPORTED_FORMATS[file_ext],
                'doc_id': doc_id,
                'namespace': self.namespace,
                'ingested_at': datetime.now().isoformat(),
                'char_count': len(text_content),
            }
            
            if org_id:
                base_metadata['org_id'] = org_id
            
            if metadata:
                base_metadata.update(metadata)
            
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                vector_id = f"{doc_id}_chunk_{i}"
                
                # Format metadata to match expected structure
                chunk_metadata = {
                    'url': f"document://{filename}",  # Use document:// protocol for docs
                    'title': filename,
                    'section': f'Chunk {i+1}',
                    'crawl_ts': datetime.now().isoformat(),
                    'site': 'document',
                    'namespace': self.namespace,
                    'chunk_index': i,
                    'source_type': 'document',
                    'file_type': self.SUPPORTED_FORMATS[file_ext],
                    'doc_id': doc_id,
                }
                
                # Add org_id if provided
                if org_id:
                    chunk_metadata['org_id'] = org_id
                
                # Add custom metadata
                if metadata:
                    chunk_metadata.update(metadata)
                
                vectors.append({
                    'id': vector_id,
                    'embedding': embedding,  # Changed from 'values' to 'embedding'
                    'text': chunk['text'],
                    'metadata': chunk_metadata
                })
            
            # Upsert to Pinecone
            print(f"  ☁️  Uploading {len(vectors)} vectors to Pinecone namespace '{self.namespace}'...")
            print(f"     Sample vector ID: {vectors[0]['id'] if vectors else 'N/A'}")
            print(f"     Embedding dimension: {len(vectors[0]['embedding']) if vectors and vectors[0].get('embedding') else 'N/A'}")
            
            success = upsert_chunks(vectors, batch_size=100, namespace=self.namespace)
            
            if not success:
                raise Exception("Failed to upsert vectors to Pinecone")
            
            print(f"  ✅ Successfully uploaded vectors to Pinecone")
            
            # Update document record to completed
            if org_id and user_id:
                update_document_success(
                    doc_id=doc_id,
                    chunk_count=len(chunks),
                    character_count=len(text_content)
                )
            
            elapsed = (datetime.now() - start_time).total_seconds()
            
            result = {
                'success': True,
                'filename': filename,
                'doc_id': doc_id,
                'chunks': len(chunks),
                'characters': len(text_content),
                'namespace': self.namespace,
                'elapsed_seconds': round(elapsed, 2),
                'vectors_uploaded': len(vectors)
            }
            
            print(f"  ✅ Successfully ingested {filename}")
            print(f"  ⏱️  Completed in {elapsed:.2f}s")
            print(f"  📊 Stats: {len(chunks)} chunks, {len(text_content)} chars, {len(vectors)} vectors")
            
            return result
            
        except Exception as e:
            import traceback
            error_msg = str(e)
            print(f"  ❌ Error ingesting document: {error_msg}")
            print(f"  📋 Traceback: {traceback.format_exc()}")
            
            # Update document record to failed
            if org_id and user_id:
                update_document_failure(doc_id, error_msg)
            
            return {
                'success': False,
                'filename': filename,
                'error': error_msg,
                'namespace': self.namespace,
                'traceback': traceback.format_exc()
            }
    
    @staticmethod
    def is_supported_format(filename: str) -> bool:
        """Check if file format is supported."""
        ext = Path(filename).suffix.lower()
        return ext in DocumentIngestor.SUPPORTED_FORMATS
    
    @staticmethod
    def get_supported_formats() -> List[str]:
        """Get list of supported file extensions."""
        return list(DocumentIngestor.SUPPORTED_FORMATS.keys())
