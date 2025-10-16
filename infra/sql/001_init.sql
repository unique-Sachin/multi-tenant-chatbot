-- Initial database schema for Zibtek chatbot
-- Creates chat_logs table and required indexes

-- Enable UUID extension if not already enabled
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Create chat_logs table
CREATE TABLE IF NOT EXISTS chat_logs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    ts TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    session_id TEXT NOT NULL,
    user_query TEXT NOT NULL,
    is_oos BOOLEAN NOT NULL DEFAULT FALSE,
    retrieved_ids JSONB,
    retrieved_urls JSONB,
    rerank_scores JSONB,
    answer TEXT NOT NULL,
    citations JSONB,
    model TEXT,
    latency_ms INTEGER,
    cost_cents INTEGER,
    flags JSONB
);

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_chat_logs_ts ON chat_logs (ts);
CREATE INDEX IF NOT EXISTS idx_chat_logs_is_oos_ts ON chat_logs (is_oos, ts);
CREATE INDEX IF NOT EXISTS idx_chat_logs_session_id ON chat_logs (session_id);

-- Add comments for documentation
COMMENT ON TABLE chat_logs IS 'Stores all chat interactions and responses from the Zibtek chatbot';
COMMENT ON COLUMN chat_logs.id IS 'Unique identifier for each chat log entry';
COMMENT ON COLUMN chat_logs.ts IS 'Timestamp when the chat interaction occurred';
COMMENT ON COLUMN chat_logs.session_id IS 'Session identifier to group related chat interactions';
COMMENT ON COLUMN chat_logs.user_query IS 'The original user query/question';
COMMENT ON COLUMN chat_logs.is_oos IS 'Whether the query was out-of-scope (not related to Zibtek)';
COMMENT ON COLUMN chat_logs.retrieved_ids IS 'JSON array of document IDs retrieved for this query';
COMMENT ON COLUMN chat_logs.retrieved_urls IS 'JSON array of URLs retrieved for this query';
COMMENT ON COLUMN chat_logs.rerank_scores IS 'JSON object of reranking scores for retrieved documents';
COMMENT ON COLUMN chat_logs.answer IS 'The chatbot response/answer';
COMMENT ON COLUMN chat_logs.citations IS 'JSON array of citations used in the response';
COMMENT ON COLUMN chat_logs.model IS 'The AI model used to generate the response';
COMMENT ON COLUMN chat_logs.latency_ms IS 'Response time in milliseconds';
COMMENT ON COLUMN chat_logs.cost_cents IS 'API cost in cents for this interaction';
COMMENT ON COLUMN chat_logs.flags IS 'JSON object for additional metadata and flags';

-- Create a view for analytics
CREATE OR REPLACE VIEW chat_analytics AS
SELECT 
    DATE_TRUNC('day', ts) AS date,
    COUNT(*) AS total_queries,
    COUNT(*) FILTER (WHERE is_oos = true) AS out_of_scope_queries,
    COUNT(*) FILTER (WHERE is_oos = false) AS in_scope_queries,
    ROUND(AVG(latency_ms), 2) AS avg_latency_ms,
    ROUND(AVG(cost_cents), 2) AS avg_cost_cents,
    COUNT(DISTINCT session_id) AS unique_sessions
FROM chat_logs
GROUP BY DATE_TRUNC('day', ts)
ORDER BY date DESC;

COMMENT ON VIEW chat_analytics IS 'Daily analytics view for chat interactions';

-- Grant permissions (adjust as needed for your setup)
-- GRANT SELECT, INSERT, UPDATE ON chat_logs TO your_app_user;
-- GRANT SELECT ON chat_analytics TO your_app_user;