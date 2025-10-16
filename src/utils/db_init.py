#!/usr/bin/env python3
"""Database initialization script for Zibtek chatbot using Supabase.

This script creates the chat_logs table and required indexes in Supabase.

Usage:
    python -m src.utils.db_init
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.storage.db import supabase, ChatLogCreate, create_chat_log

# Load environment variables
load_dotenv()


def check_supabase_connection() -> bool:
    """Check if Supabase client is properly initialized."""
    if not supabase:
        print("❌ Error: Supabase client not initialized")
        print("Please set SUPABASE_URL and SUPABASE_ANON_KEY in your .env file")
        return False
    
    print("✅ Supabase client is initialized")
    return True


def create_table_manually() -> bool:
    """Guide user to create table manually in Supabase dashboard."""
    print("\n📋 To create the chat_logs table in Supabase:")
    print("1. Go to https://supabase.com/dashboard")
    print("2. Select your project")
    print("3. Go to SQL Editor")
    print("4. Run the following SQL:")
    
    sql_content = read_sql_file()
    if sql_content:
        print("\n" + "="*60)
        print(sql_content)
        print("="*60)
        return True
    return False


def read_sql_file() -> str:
    """Read the SQL migration file."""
    try:
        project_root = Path(__file__).parent.parent.parent
        sql_file = project_root / "infra" / "sql" / "001_init.sql"
        
        with open(sql_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        print(f"✅ SQL file read successfully: {sql_file}")
        return content
    except FileNotFoundError:
        project_root = Path(__file__).parent.parent.parent
        sql_file = project_root / "infra" / "sql" / "001_init.sql"
        print(f"❌ Error: SQL file not found at {sql_file}")
        return ""
    except Exception as e:
        print(f"❌ Error reading SQL file: {e}")
        return ""


def test_table_exists() -> bool:
    """Test if the chat_logs table exists and is accessible."""
    if not supabase:
        print("❌ Supabase client not available for testing")
        return False
        
    try:
        print("🧪 Testing if chat_logs table exists...")
        
        # Try to query the table (will fail if table doesn't exist)
        result = supabase.table("chat_logs").select("id").limit(1).execute()
        
        if hasattr(result, 'data'):
            print("✅ chat_logs table exists and is accessible")
            return True
        else:
            print("❌ Unexpected response format")
            return False
            
    except Exception as e:
        print(f"❌ Table test failed: {e}")
        if "Could not find the table" in str(e):
            print("   → Table doesn't exist yet. Create it using the SQL above.")
        return False


def test_crud_operations() -> bool:
    """Test basic CRUD operations on the chat_logs table."""
    try:
        print("🧪 Testing CRUD operations...")
        
        # Create a test chat log
        test_log = ChatLogCreate(
            session_id="init-test-session",
            user_query="Test query during database initialization",
            answer="Test answer to verify database functionality",
            is_oos=False,
            model="test-model",
            latency_ms=100,
            cost_cents=1
        )
        
        # Test CREATE
        created_log = create_chat_log(test_log)
        if not created_log:
            print("❌ Failed to create test log")
            return False
        
        print(f"✅ CREATE: Test log created with ID {created_log.get('id', 'unknown')}")
        
        # Test READ - get recent logs
        from src.storage.db import get_chat_logs
        logs = get_chat_logs(limit=5)
        print(f"✅ READ: Retrieved {len(logs)} recent logs")
        
        # Test STATS
        from src.storage.db import get_chat_log_stats
        stats = get_chat_log_stats()
        print(f"✅ STATS: {stats}")
        
        return True
        
    except Exception as e:
        print(f"❌ CRUD operations test failed: {e}")
        return False


def main():
    """Main function to initialize and test Supabase database."""
    print("🚀 Initializing Zibtek chatbot database with Supabase...")
    print("=" * 60)
    
    # Step 1: Check Supabase connection
    if not check_supabase_connection():
        sys.exit(1)
    
    # Step 2: Check if table exists
    table_exists = test_table_exists()
    
    if not table_exists:
        print("\n� Table creation required!")
        create_table_manually()
        print("\n⏳ After creating the table, run this script again to test functionality.")
        sys.exit(0)
    
    # Step 3: Test CRUD operations
    print("\n🧪 Testing database functionality...")
    if test_crud_operations():
        print("\n" + "=" * 60)
        print("🎉 Database initialization and testing complete!")
        print("\nYour Supabase database is ready for the chatbot!")
        print("\nNext steps:")
        print("1. Start building your data ingestion pipeline")
        print("2. Create retrieval and guard modules")
        print("3. Build the chat interface")
    else:
        print("\n❌ Database functionality tests failed")
        print("Check your Supabase configuration and table structure")
        sys.exit(1)


if __name__ == "__main__":
    main()