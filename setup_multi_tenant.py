"""Initialize database with multi-tenant schema.

Run this script to set up the organizations, websites, and ingestion_jobs tables.
"""

import os
import sys
from dotenv import load_dotenv

# Add src to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.storage.db import supabase

load_dotenv()


def run_migration():
    """Run the multi-tenant migration."""
    if not supabase:
        print("❌ Supabase client not initialized. Set SUPABASE_URL and SUPABASE_ANON_KEY in .env")
        return False
    
    # Read SQL migration file
    sql_file = os.path.join(os.path.dirname(__file__), '..', 'infra', 'sql', '002_multi_tenant.sql')
    
    if not os.path.exists(sql_file):
        print(f"❌ Migration file not found: {sql_file}")
        return False
    
    with open(sql_file, 'r') as f:
        sql = f.read()
    
    print("📦 Running multi-tenant migration...")
    print("=" * 60)
    
    try:
        # Note: Supabase Python client doesn't support raw SQL execution
        # Users need to run this in the Supabase SQL Editor
        print("⚠️  Please run the migration manually in Supabase SQL Editor:")
        print()
        print("1. Go to: https://supabase.com/dashboard")
        print("2. Select your project")
        print("3. Go to SQL Editor")
        print("4. Copy and paste the contents of:")
        print(f"   {sql_file}")
        print("5. Click 'Run'")
        print()
        print("Or connect via psql:")
        print(f"   psql -h <your-host> -U postgres -d postgres -f {sql_file}")
        print()
        
        # Test if tables exist
        print("🔍 Checking if tables already exist...")
        
        try:
            # Try to query organizations table
            result = supabase.table("organizations").select("*").limit(1).execute()
            print("✅ organizations table exists")
            
            result = supabase.table("websites").select("*").limit(1).execute()
            print("✅ websites table exists")
            
            result = supabase.table("ingestion_jobs").select("*").limit(1).execute()
            print("✅ ingestion_jobs table exists")
            
            print()
            print("🎉 All tables found! Migration already applied.")
            return True
            
        except Exception as e:
            print(f"⚠️  Tables not found: {e}")
            print()
            print("Please run the migration SQL manually (see instructions above)")
            return False
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def main():
    """Main function."""
    print("🚀 Multi-Tenant Database Setup")
    print("=" * 60)
    
    success = run_migration()
    
    if success:
        print()
        print("✨ Database is ready for multi-tenant ingestion!")
        print()
        print("Next steps:")
        print("1. Start the API server: uvicorn src.app.server:app --reload")
        print("2. Start the UI: streamlit run src/ui/app.py")
        print("3. Go to http://localhost:8501")
        print("4. Navigate to 🏢 Organizations tab")
        print("5. Create your first organization!")
    else:
        print()
        print("❌ Setup incomplete. Please follow the instructions above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
