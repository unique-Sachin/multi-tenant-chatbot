-- User authentication and conversation history tables

-- Users table
CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    full_name VARCHAR(255),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    last_login_at TIMESTAMP WITH TIME ZONE,
    is_active BOOLEAN DEFAULT true
);

-- Create index on email for faster lookups
CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);

-- Update chat_logs table to include user_id, namespace, and retrieval_steps
ALTER TABLE chat_logs 
ADD COLUMN IF NOT EXISTS user_id UUID REFERENCES users(id) ON DELETE CASCADE,
ADD COLUMN IF NOT EXISTS namespace VARCHAR(255) DEFAULT 'zibtek',
ADD COLUMN IF NOT EXISTS retrieval_steps JSONB;

-- Create index on user_id and namespace for faster conversation retrieval
CREATE INDEX IF NOT EXISTS idx_chat_logs_user_namespace ON chat_logs(user_id, namespace);
CREATE INDEX IF NOT EXISTS idx_chat_logs_user_session ON chat_logs(user_id, session_id);

-- Conversation sessions table (for better organization)
CREATE TABLE IF NOT EXISTS conversation_sessions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    namespace VARCHAR(255) NOT NULL,
    session_id VARCHAR(255) NOT NULL,
    title VARCHAR(500),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(user_id, namespace, session_id)
);

-- Create indexes
CREATE INDEX IF NOT EXISTS idx_conv_sessions_user ON conversation_sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_conv_sessions_namespace ON conversation_sessions(user_id, namespace);

-- Function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Trigger to auto-update updated_at
DROP TRIGGER IF EXISTS update_conversation_sessions_updated_at ON conversation_sessions;
CREATE TRIGGER update_conversation_sessions_updated_at
    BEFORE UPDATE ON conversation_sessions
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Comments
COMMENT ON TABLE users IS 'User accounts for authentication';
COMMENT ON TABLE conversation_sessions IS 'Conversation sessions organized by user and namespace';
COMMENT ON COLUMN chat_logs.user_id IS 'Reference to the user who created this message';
COMMENT ON COLUMN chat_logs.namespace IS 'Organization namespace for multi-tenant isolation';
