-- Multi-tenant schema for organization-based ingestion
-- Allows multiple organizations to manage their own websites and knowledge bases

-- Organizations table
CREATE TABLE IF NOT EXISTS organizations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name TEXT NOT NULL,
    slug TEXT NOT NULL UNIQUE,
    description TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Websites table (each organization can have multiple websites)
CREATE TABLE IF NOT EXISTS websites (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    org_id UUID NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
    url TEXT NOT NULL,
    namespace TEXT NOT NULL UNIQUE,
    status TEXT NOT NULL DEFAULT 'pending',
    pages_crawled INTEGER DEFAULT 0,
    chunks_created INTEGER DEFAULT 0,
    last_ingested_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Ingestion jobs table (track ingestion progress)
CREATE TABLE IF NOT EXISTS ingestion_jobs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    website_id UUID NOT NULL REFERENCES websites(id) ON DELETE CASCADE,
    status TEXT NOT NULL DEFAULT 'pending',
    progress_percent INTEGER DEFAULT 0,
    pages_crawled INTEGER DEFAULT 0,
    chunks_created INTEGER DEFAULT 0,
    error_message TEXT,
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Create indexes
CREATE INDEX IF NOT EXISTS idx_websites_org_id ON websites (org_id);
CREATE INDEX IF NOT EXISTS idx_websites_namespace ON websites (namespace);
CREATE INDEX IF NOT EXISTS idx_ingestion_jobs_website_id ON ingestion_jobs (website_id);
CREATE INDEX IF NOT EXISTS idx_ingestion_jobs_status ON ingestion_jobs (status);

-- Add comments
COMMENT ON TABLE organizations IS 'Organizations that own websites and knowledge bases';
COMMENT ON TABLE websites IS 'Websites to crawl and ingest per organization';
COMMENT ON TABLE ingestion_jobs IS 'Track ingestion job progress and status';

COMMENT ON COLUMN websites.namespace IS 'Unique Pinecone namespace for this website';
COMMENT ON COLUMN websites.status IS 'Status: pending, ingesting, completed, failed';
COMMENT ON COLUMN ingestion_jobs.status IS 'Job status: pending, running, completed, failed';

-- Create a view for organization stats
CREATE OR REPLACE VIEW organization_stats AS
SELECT 
    o.id AS org_id,
    o.name AS org_name,
    o.slug AS org_slug,
    COUNT(DISTINCT w.id) AS total_websites,
    SUM(w.pages_crawled) AS total_pages_crawled,
    SUM(w.chunks_created) AS total_chunks_created,
    MAX(w.last_ingested_at) AS last_ingestion_date
FROM organizations o
LEFT JOIN websites w ON w.org_id = o.id
GROUP BY o.id, o.name, o.slug;

-- Insert default Zibtek organization for existing data
INSERT INTO organizations (name, slug, description)
VALUES ('Zibtek', 'zibtek', 'Zibtek software development company')
ON CONFLICT (slug) DO NOTHING;

-- Insert default Zibtek website
INSERT INTO websites (org_id, url, namespace, status, last_ingested_at)
SELECT 
    o.id,
    'https://www.zibtek.com',
    'zibtek',
    'completed',
    NOW()
FROM organizations o
WHERE o.slug = 'zibtek'
ON CONFLICT (namespace) DO NOTHING;
