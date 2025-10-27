# Multi-Tenant AI Chatbot System

A production-grade, **multi-tenant** AI chatbot system that allows multiple organizations to manage their own websites, trigger custom ingestion, and chat with organization-specific knowledge bases. Each organization has completely isolated data and can customize their chatbot experience.

## 🌟 New: Multi-Tenant Support

The system now supports **multiple organizations**, each with isolated knowledge bases:

- 🏢 **Create Organizations**: Add multiple companies/teams
- 🌐 **Add Websites**: Each organization can have multiple websites
- 🚀 **Trigger Ingestion**: On-demand website crawling and vectorization
- 💬 **Namespace Isolation**: Each organization's data is completely separate
- 📊 **Progress Tracking**: Real-time ingestion job monitoring

**Multi-Tenant Setup:** Use the SQL scripts in `infra/sql/` to initialize the database schema for organizations and websites. The system uses Milvus partitions for data isolation.

## Features

- **Multi-Tenant Architecture**: Isolated knowledge bases per organization using Milvus partitions
- **Organization Management**: Unlimited organizations, each with their own websites and knowledge bases
- **Dynamic Ingestion**: Add and crawl new websites through the UI
- **Custom Dataset Integration**: Each organization responds using only their website data
- **Out-of-Scope Handling**: Politely rejects questions unrelated to organization's content
- **Prompt Injection Protection**: Implements security guardrails
- **Comprehensive Logging**: Records all interactions with organization isolation
- **Multiple Retrieval Methods**: Embeddings, vector search, and keyword matching
- **Modern Web UI**: Streamlit interface with organization management and chat
- **Real-time Citations**: Clickable source links with each response
- **Session Management**: Persistent chat history and user sessions
- **Background Jobs**: Async ingestion with progress tracking
- **Document Upload**: PDF, TXT, and DOCX file upload and processing per organization
- **User Authentication**: JWT-based user registration and login
- **Conversation History**: Persistent chat sessions with context-aware follow-ups
- **Hybrid Search**: Vector + BM25 keyword search with RRF fusion
- **Performance Caching**: Redis-based caching for improved response times
- **RAGAS Evaluation**: Comprehensive evaluation framework with detailed metrics

## Architecture

```
Frontend (Streamlit UI) → FastAPI Server → Guardrails → Retrieval → LLM → Database
```

- **Streamlit UI**: Authentication, chat interface, streaming responses
- **FastAPI Server**: REST API with async endpoints and health monitoring
- **Guardrails**: Multi-layer content filtering and injection protection
- **Retrieval**: Milvus vector search with semantic filtering
- **LLM**: OpenAI GPT-4o-mini for response generation
- **Database**: Supabase PostgreSQL for interaction logging

## Project Structure

```
multitenant-chatbot/
├── src/
│   ├── app/          # FastAPI server
│   │   ├── server.py # Main API endpoints + organization endpoints
│   │   ├── jobs.py   # Background job manager
│   │   └── ingestion_runner.py # Ingestion with progress tracking
│   ├── ingest/       # Data collection and processing
│   │   ├── ingest.py # Website crawling and vectorization (multi-tenant)
│   │   └── document_ingest.py # Document upload processing
│   ├── retrieval/    # Search and retrieval systems
│   │   ├── retriever.py # Milvus + LangChain integration
│   │   └── rerank.py    # Cohere reranking
│   ├── guards/       # Security and scope validation
│   │   └── guards.py # Multi-layer safety system
│   ├── storage/      # Database and vector store interfaces
│   │   ├── db.py     # Supabase database operations
│   │   ├── milvus.py # Milvus vector storage (partition support)
│   │   ├── organizations.py # Organization/website CRUD
│   │   ├── auth.py   # User authentication
│   │   ├── conversations.py # Chat history management
│   │   └── documents.py # Document management
│   ├── ui/           # Streamlit frontend
│   │   ├── app.py    # Main chat interface with namespace selector
│   │   └── pages/
│   │       ├── organizations.py # Organization management UI
│   │       └── documents.py # Document upload UI

│   └── utils/        # Shared utilities
│       ├── db_init.py # Database initialization
│       ├── cache.py  # Redis caching
│       └── retries.py # Retry logic
├── evaluation/       # RAGAS evaluation framework
│   ├── README.md     # Evaluation documentation
│   ├── ragas_evaluator.py # RAGAS evaluation system
│   ├── ragas_analysis.py  # Analysis utilities
│   ├── run_ragas.py      # Run evaluations
│   └── test_cases.json   # Test cases for evaluation
├── infra/
│   ├── docker/       # Docker configurations
│   │   ├── docker-compose.yml # Multi-service setup
│   │   └── Dockerfile        # Container definition
│   └── sql/          # Database schemas
│       ├── 001_init.sql         # Initial chat_logs table
│       ├── 002_multi_tenant.sql # Multi-tenant tables
│       ├── 002_users_and_conversations.sql # User auth and conversations
│       └── 003_documents_table.sql # Document management
├── deploy-heroku.sh      # Heroku deployment script
├── pyproject.toml        # Project configuration
├── requirements.txt      # Core dependencies
├── requirements-dev.txt  # Development dependencies
├── Procfile             # Heroku process definition
├── .env.example         # Environment variables template
└── .slugignore          # Heroku deployment exclusions
```

## Quick Start

### 1. Environment Setup

```bash
cd multitenant-chatbot
cp .env.example .env
```

Edit `.env` with your API keys:
```env
SUPABASE_URL=your_supabase_url_here
SUPABASE_ANON_KEY=your_supabase_anon_key_here
MILVUS_URI=your_milvus_uri_here
MILVUS_TOKEN=your_milvus_token_here
OPENAI_API_KEY=your_openai_api_key_here
```

### 2. Install Dependencies

```bash
# Using virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Or using uv (if available)
uv sync
```

### 3. Initialize Database

```bash
python src/utils/db_init.py
```

### 4. Setup Multi-Tenant Schema

Run the SQL scripts in order to set up the multi-tenant database schema in Supabase:

1. `infra/sql/001_init.sql` - Initial tables
2. `infra/sql/002_multi_tenant.sql` - Organization tables
3. `infra/sql/002_users_and_conversations.sql` - User authentication
4. `infra/sql/003_documents_table.sql` - Document management

### 5. Run the Complete Chat System

**Start the API server:**
```bash
uvicorn src.app.server:app --host 0.0.0.0 --port 8000 --reload
```

**Start the Streamlit UI (in a new terminal):**
```bash
streamlit run src/ui/app.py --server.port 8501 --server.address 0.0.0.0
```

#### Access the Application

- **Streamlit UI**: http://localhost:8501
- **API Documentation**: http://localhost:8000/docs
- **API Health**: http://localhost:8000/health

**Authentication:** 
- Sign up with email/password on the Streamlit UI
- JWT-based authentication with refresh tokens
- User sessions persist across browser restarts

## 🐳 Docker Deployment (One Command)

For production or simplified local development, use Docker Compose:

### Prerequisites
- Docker and Docker Compose installed
- `.env` file configured with your API keys

### Quick Start with Docker

**Docker Setup**
```bash
# Ensure .env file exists with your API keys
cp .env.example .env
# Edit .env with your credentials

# Build and run with Docker
docker build -f infra/docker/Dockerfile -t multi-tenant-chatbot .
docker run -p 8000:8000 --env-file .env multi-tenant-chatbot
```

Note: A docker-compose.yml file can be created for multi-service deployment if needed.

### Docker Usage

The Docker container runs the FastAPI backend server. You'll need to run Streamlit separately:

```bash
# Run the API container
docker run -p 8000:8000 --env-file .env multi-tenant-chatbot

# In another terminal, run Streamlit locally
streamlit run src/ui/app.py --server.port 8501
```

### Access URLs
- **API Docs**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health
- **Chat Interface**: http://localhost:8501 (when running Streamlit locally)

### Docker Commands

```bash
# Start services in background
docker compose -f infra/docker/docker-compose.yml up -d --build

# View logs
docker compose -f infra/docker/docker-compose.yml logs -f

# Stop services
docker compose -f infra/docker/docker-compose.yml down

# Restart a specific service
docker compose -f infra/docker/docker-compose.yml restart api

# Check service health
docker compose -f infra/docker/docker-compose.yml ps
```

### Docker Environment

The Docker setup automatically:
- Installs all Python dependencies
- Configures networking between services
- Sets up health checks for reliability
- Mounts your `.env` file for configuration
- Initializes PostgreSQL with schema
- Provides persistent data volumes

### Access URLs (Docker)
- **Chat Interface**: http://localhost:8501
- **API Docs**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health
- **Redis**: localhost:6379
- **PostgreSQL**: localhost:5432 (user: zibtek, db: zibtek)

## 🚀 Heroku Deployment

For production deployment to Heroku:

### Prerequisites
- Heroku CLI installed
- Heroku account
- External Supabase and Milvus accounts (Heroku doesn't include these)

### Quick Deploy
```bash
# Make script executable and run
chmod +x deploy-heroku.sh
./deploy-heroku.sh
```

The script will:
1. Create or use existing Heroku app
2. Add Redis addon
3. Set environment variables from `.env`
4. Deploy the API server
5. Provide health check and management commands

### Manual Deployment
```bash
# Login to Heroku
heroku login

# Create app
heroku create your-app-name

# Add Redis
heroku addons:create heroku-redis:mini

# Set environment variables
heroku config:set OPENAI_API_KEY=your_key
heroku config:set MILVUS_URI=your_uri
heroku config:set MILVUS_TOKEN=your_token
# ... (set all required variables)

# Deploy
git push heroku main
```

### Production Notes
- Only the API is deployed to Heroku (not Streamlit UI)
- Use external Supabase for database
- Configure CORS for your frontend domain
- Monitor with `heroku logs --tail`

### 7. Features Overview

**Streamlit UI Features:**
- 🔐 Shared-secret authentication gate
- 💬 Real-time chat with streaming responses
- 📚 Clickable citation chips for sources
- ⏱️ Response latency and status indicators
- 🔄 Session management and chat history
- 💡 Example questions in sidebar
- 🟢 Green for grounded responses, 🟡 amber for refusals

**Backend Features:**
- 🛡️ Multi-layer guardrails (keyword + semantic)
- 🔍 Pinecone vector search with LangChain
- 🤖 GPT-4o-mini for response generation
- 📊 Complete interaction logging to Supabase
- 🚀 Async FastAPI with health monitoring

## Data Ingestion Pipeline

The ingestion pipeline (`src/ingest/ingest.py`) performs the following steps:

### 1. URL Discovery
- Attempts to find and parse `sitemap.xml`
- Filters URLs to same domain only
- Excludes non-content files (PDFs, images, etc.)

### 2. Content Extraction
- Uses Trafilatura for high-quality text extraction
- Falls back to BeautifulSoup for problematic pages
- Extracts page titles and main content

### 3. Content Chunking
- Splits content into ~1000 token chunks
- Maintains 200 token overlap between chunks
- Preserves sentence boundaries
- Generates deterministic chunk IDs

### 4. Embedding Generation
- Uses OpenAI's `text-embedding-3-small` model
- Processes chunks in batches to respect rate limits
- Handles API errors gracefully

### 5. Vector Storage
- Stores chunks with embeddings in Milvus
- Includes metadata (URL, title, section, timestamp)
- Uses deterministic IDs for deduplication
- Supports partition-based multi-tenancy

### 6. Run the Complete Chat System

**Start the API server:**
```bash
uvicorn src.app.server:app --host 0.0.0.0 --port 8000 --reload
```

**Start the Streamlit UI (in a new terminal):**
```bash
streamlit run src/ui/app.py --server.port 8501 --server.address 0.0.0.0
```

## Retrieval System

## Document Upload & Processing

The system supports uploading and processing various document formats:

### Supported Formats
- **PDF**: Portable Document Format files
- **TXT**: Plain text files
- **DOCX**: Microsoft Word documents

### Upload Process
1. **Authentication**: User must be logged in
2. **Organization Selection**: Choose target organization
3. **File Upload**: Drag & drop or browse for files
4. **Processing**: Automatic text extraction and chunking
5. **Vectorization**: Generate embeddings for search
6. **Storage**: Save to organization's partition in Milvus

### Access Document Upload
- Navigate to 📄 **Documents** page in the Streamlit UI
- Select your organization
- Upload files up to 10MB
- Monitor processing progress

## User Authentication & Sessions

The system includes a complete authentication system:

### Features
- **User Registration**: Email/password signup
- **Secure Login**: JWT-based authentication
- **Session Persistence**: Login state maintained across browser sessions
- **Conversation History**: Chat history tied to user accounts with context awareness
- **Organization Access**: Users can access multiple organizations

### Conversation History & Context
- **Context-Aware Follow-ups**: Follow-up questions automatically include previous 3 Q&A pairs as context
- **Session Continuity**: Each conversation session maintains its own history thread
- **Extended Context**: Last 3 relevant conversations provide better context understanding
- **Multi-tenant Support**: Conversation history is isolated by organization namespace

### Usage
1. Visit the Streamlit UI
2. Sign up with email and password
3. Login to access the chat interface
4. Your conversations are automatically saved
5. Ask follow-up questions that reference previous responses
6. Each organization maintains separate conversation contexts

## Evaluation Framework

A comprehensive RAGAS-based evaluation system is included:

### Features
- **Multiple Metrics**: Faithfulness, Answer Relevancy, Context Precision, Context Recall
- **Test Cases**: Predefined test questions and expected responses
- **Automated Reports**: HTML reports with detailed analysis
- **Continuous Monitoring**: Regular evaluation of system performance

### Usage
```bash
cd evaluation
python run_ragas.py
```

See `evaluation/README.md` for detailed documentation.

## Retrieval System

The retrieval system (`src/retrieval/retriever.py`) provides semantic search capabilities:

### Key Features
- **Milvus Integration**: Uses Milvus vector database for high-performance retrieval
- **Score-based Filtering**: Optional similarity score thresholds
- **Metadata Filtering**: Filter by site, URL patterns, or custom metadata
- **Multi-tenant Support**: Partition-based isolation for organizations
- **Hybrid Search**: Vector similarity with BM25 keyword matching

### Usage Examples

```python
from src.retrieval.retriever import retrieve

# Simple retrieval (organization-specific)
docs = retrieve("What services does our company offer?", k=5, partition_name="org_123")

# Advanced retrieval with scores and filtering
docs = retrieve(
    "software development", 
    k=3, 
    partition_name="org_123",
    score_threshold=0.7
)
```

### Test Coverage
- Unit tests with mocked dependencies
- Integration tests with real Milvus API
- Multi-tenant partition isolation validation
- Validates non-empty results and proper metadata structure

## Guardrails System

The guardrails system (`src/guards/guards.py`) provides multi-layer safety before LLM processing:

### Multi-Layer Safety Architecture

**Layer 1: Hard Keyword Filter**
- Regex-based blocking of clearly out-of-scope topics
- Keywords: `president|iphone|bitcoin|weather|cricket|football|movie|stock price`
- Fast, deterministic blocking of inappropriate queries

**Layer 2: Semantic Scope Validation**
- Computes corpus centroid from 500 random Milvus vectors
- Calculates cosine similarity between question and corpus centroid
- Configurable similarity threshold via `MIN_SCOPE_SIM` environment variable
- Blocks questions semantically distant from organization content
- Supports multi-tenant partition-based checking

**Layer 3: Content Sanitization**
- Removes `<script>` blocks and harmful HTML tags
- Strips prompt injection patterns: "ignore previous...", "act as...", "you are now..."
- Neutralizes attempts to change assistant behavior
- Preserves legitimate content while removing malicious elements

**Layer 4: Strict System Prompt**
- Enforces organization-specific responses using provided CONTEXT
- Requires URL citations for all information
- Explicit out-of-scope message for questions unrelated to organization's content
- Hardened against prompt injection and roleplay attempts

### Usage Examples

```python
from src.guards.guards import ZibtekGuards, is_out_of_scope, sanitize, system_prompt

# Complete validation pipeline
guards = ZibtekGuards()
result = guards.validate_question("What services does our company offer?")

# Individual functions
is_blocked, reason = is_out_of_scope("Who is the president?", "org_123")  # Returns (True, "Hard keyword filter")
clean_text = sanitize("<script>alert('xss')</script>Hello")    # Returns "Hello"
prompt = system_prompt()                                        # Returns complete system prompt
```

### Configuration Options

- `MIN_SCOPE_SIM`: Minimum cosine similarity threshold (default: 0.4)
- `OUT_OF_SCOPE_MESSAGE`: Custom message for blocked questions
- Corpus centroid caching per organization (24-hour cache duration)

## Configuration

### Environment Variables

| Variable | Description | Required | Default |
|----------|-------------|----------|----------|
| `SUPABASE_URL` | Supabase project URL | Yes | - |
| `SUPABASE_ANON_KEY` | Supabase anonymous key | Yes | - |
| `MILVUS_URI` | Milvus server URI | Yes | - |
| `MILVUS_TOKEN` | Milvus access token | Yes | - |
| `OPENAI_API_KEY` | OpenAI API key | Yes | - |
| `MILVUS_COLLECTION` | Milvus collection name | No | documents |
| `COHERE_API_KEY` | Cohere API key for reranking | No | - |
| `REDIS_URL` | Redis connection URL | No | redis://localhost:6379 |
| `MIN_SCOPE_SIM` | Minimum similarity threshold for scope validation | No | 0.4 |
| `OUT_OF_SCOPE_MESSAGE` | Custom message for blocked questions | No | (default message) |
| `JWT_SECRET` | JWT token signing secret | Yes | (auto-generated) |
| `API_BASE_URL` | API base URL for Streamlit | No | http://localhost:8000 |
| `RERANK_ENABLED` | Enable Cohere reranking | No | true |
| `CACHE_ENABLED` | Enable Redis caching | No | false |
| `RATELIMIT_ENABLED` | Enable API rate limiting | No | true |

### Ingestion Settings

You can modify these settings in `src/ingest/ingest.py`:

```python
self.chunk_size = 1000      # tokens per chunk
self.chunk_overlap = 200    # token overlap
self.max_pages = 500         # limit for testing (per organization)
```

### Guardrails Settings

```env
MIN_SCOPE_SIM=0.4  # Minimum similarity threshold
OUT_OF_SCOPE_MESSAGE="Custom message for blocked questions"
```

## FastAPI Server

The main chatbot server (`src/app/server.py`) provides a complete REST API:

### Starting the Server
```bash
python src/app/server.py
```
- **API**: http://localhost:8000
- **Docs**: http://localhost:8000/docs
- **Health**: http://localhost:8000/health

### API Endpoints

**POST /chat/stream** - Main chat endpoint
```bash
curl -X POST http://localhost:8000/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"question": "What services does your company offer?", "namespace": "org_123"}'
```

**GET /health** - Service health check  

### Complete Pipeline
1. **Guardrails**: Multi-layer safety validation
2. **Retrieval**: Top-4 from 20 Milvus results  
3. **Context**: LLM prompt with document blocks
4. **Generation**: GPT-4o-mini with citations
5. **Logging**: Performance and cost tracking

### Testing
```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run the API server for testing
uvicorn src.app.server:app --reload

# Test API endpoints
curl http://localhost:8000/health
```

## Technologies Used
```

3. **Data Ingestion**
   ```bash
   # Add organizations and websites through the UI
   # Or use the API endpoints for organization management
   ```

4. **Run the Application**
   ```bash
   # Start Streamlit UI
   streamlit run src/ui/app.py
   
   # Or start FastAPI backend
   uvicorn src.app.server:app --reload
   ```

## Development

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run the application in development mode
uvicorn src.app.server:app --reload
streamlit run src/ui/app.py

# Check code quality (if tools are available)
# black src/
# mypy src/
```

## Key Features Summary

- ✅ **Multi-Organization Support**: Unlimited organizations with isolated data
- ✅ **Custom Data Integration**: Each organization manages their own websites and documents
- ✅ **Polite Out-of-Scope Handling**: Context-aware rejection of irrelevant questions
- ✅ **Prompt Injection Protection**: Multi-layer security guardrails
- ✅ **Comprehensive Logging**: Organization-specific interaction tracking
- ✅ **Advanced Retrieval**: Hybrid search with vector + keyword matching
- ✅ **Flexible Dataset Support**: Web scraping + document upload for any organization

## Technologies Used

- **Backend**: FastAPI with async support
- **Frontend**: Streamlit with multi-page interface
- **Vector Store**: Milvus with partition isolation for multi-tenancy
- **Database**: PostgreSQL (Supabase) for logging, user management, and organization data
- **Authentication**: JWT tokens with bcrypt password hashing
- **Caching**: Redis for performance optimization
- **Embeddings**: OpenAI text-embedding-3-small
- **LLM**: OpenAI gpt-4o-mini for response generation
- **Reranking**: Cohere for result reranking
- **Web Scraping**: Trafilatura with BeautifulSoup fallback
- **Search**: Hybrid search (Vector + BM25) with RRF fusion
- **Document Processing**: PyPDF2, python-docx for file processing
- **Evaluation**: RAGAS framework for comprehensive metrics
- **Security**: Multi-layer guardrails and prompt injection protection
- **Deployment**: Docker containerization, Heroku-ready configuration

## Use Cases

This system is perfect for:
- **SaaS Companies**: Provide AI chatbots to multiple client organizations
- **Enterprise**: Deploy separate chatbots for different departments or subsidiaries
- **Agencies**: Manage chatbots for multiple clients from a single platform
- **Educational Institutions**: Separate knowledge bases for different schools or departments
- **E-commerce Platforms**: Individual chatbots for multiple vendor stores

Each organization gets:
- Isolated knowledge base
- Custom website ingestion
- Private document uploads
- Independent conversation history
- Separate analytics and logging