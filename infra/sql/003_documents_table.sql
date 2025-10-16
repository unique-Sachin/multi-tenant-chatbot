-- Documents tracking table for uploaded files

-- Create documents table
CREATE TABLE IF NOT EXISTS documents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    org_id UUID NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    filename VARCHAR(255) NOT NULL,
    file_type VARCHAR(50) NOT NULL,
    file_size_bytes INTEGER NOT NULL,
    doc_id VARCHAR(50) UNIQUE NOT NULL,
    namespace VARCHAR(255) NOT NULL,
    chunk_count INTEGER DEFAULT 0,
    character_count INTEGER DEFAULT 0,
    status VARCHAR(50) DEFAULT 'pending',
    error_message TEXT,
    uploaded_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    processed_at TIMESTAMP WITH TIME ZONE,
    metadata JSONB
);

-- Create indexes for faster lookups
CREATE INDEX IF NOT EXISTS idx_documents_org_id ON documents(org_id);
CREATE INDEX IF NOT EXISTS idx_documents_user_id ON documents(user_id);
CREATE INDEX IF NOT EXISTS idx_documents_namespace ON documents(namespace);
CREATE INDEX IF NOT EXISTS idx_documents_doc_id ON documents(doc_id);
CREATE INDEX IF NOT EXISTS idx_documents_status ON documents(status);
CREATE INDEX IF NOT EXISTS idx_documents_uploaded_at ON documents(uploaded_at DESC);

-- Comments
COMMENT ON TABLE documents IS 'Tracks uploaded documents (PDF, TXT, DOCX) and their ingestion status';
COMMENT ON COLUMN documents.doc_id IS 'SHA256 hash of file content (first 16 chars)';
COMMENT ON COLUMN documents.status IS 'Status: pending, processing, completed, failed';
COMMENT ON COLUMN documents.namespace IS 'Pinecone namespace where vectors are stored';
COMMENT ON COLUMN documents.chunk_count IS 'Number of chunks created from the document';
COMMENT ON COLUMN documents.character_count IS 'Total characters extracted from document';
COMMENT ON COLUMN documents.metadata IS 'Additional metadata (title, tags, description, etc.)';
