# Zibtek AI Chatbot – System Architecture

A production-grade RAG chatbot that answers questions strictly about Zibtek using a layered architecture with guardrails, hybrid retrieval, reranking, caching, and complete logging.

## Overview

- Scope: Zibtek-only knowledge, rejects out-of-scope queries
- Stack: FastAPI + Streamlit + Pinecone + OpenAI + Cohere + Supabase + Redis
- Features: Multi-layer security, Hybrid retrieval (Vector + BM25), Cohere rerank, Streaming UI, Caching, Full logging, Evaluation framework

## High-level Data Flow

```
User → Streamlit UI → FastAPI (/chat | /chat/stream)
      → Guardrails (keywords + semantic scope)
      → Retrieval (Pinecone vector + BM25 keyword) → RRF Fusion → Cohere Rerank (top-N)
      → LLM Generation (OpenAI)
      → Supabase Log + Redis Cache → Response + Citations
```

## Components

### 1) UI: Streamlit Frontend (`src/ui/app.py`)
- Authentication: shared secret via STREAMLIT_AUTH_SHARED_SECRET
- Chat UI: message history, streaming responses
- API Health: checks `/health`
- Session: UUID-based sessions persisted in session_state

### 2) API: FastAPI Server (`src/app/server.py`)
- Endpoints:
  - `GET /health`: readiness and config status
  - `POST /chat`: synchronous chat (full response)
  - `POST /chat/stream`: streaming (SSE)
- Cross-cutting concerns:
  - Rate limiting (SlowAPI) – per-IP limits
  - Request validation (Pydantic models)
  - Caching (Redis): retrieval and final answer caches
  - Retry logic for external APIs
  - Cost + latency tracking (tiktoken token counting)

### 3) Security: Guardrails (`src/guards/guards.py`)
- Layer 1: Hard keyword filter (regex for politics, news, weather, crypto, etc.)
- Layer 2: Semantic scope check against corpus centroid (cosine similarity threshold via `MIN_SCOPE_SIM`, default 0.5)
- Returns a polite, constrained out-of-scope message on failure
- Defends against prompt injection and jailbreaks

### 4) Retrieval Engine (`src/retrieval/`)

#### a) Vector Retrieval (`retriever.py`)
- Vector DB: Pinecone (cosine)
- Embeddings: OpenAI `text-embedding-3-small` (1536-d)
- Namespace: `zibtek`; Filter: `site=https://www.zibtek.com`

#### b) Hybrid + Fusion (`hybrid.py`)
- BM25 keyword search (rank-bm25)
- Vector search (Pinecone)
- Reciprocal Rank Fusion (RRF): combines ranks; robust to noisy scores
- Redis-cached BM25 index for fast startup
- Fallbacks: vector-only or bm25-only when needed

#### c) Reranking (`rerank.py`)
- Cohere Rerank v3.5 (cross-encoder)
- Inputs: query + candidate texts → relevance scores
- Outputs: top-N reranked documents with scores

### 5) Ingestion Pipeline (`src/ingest/ingest.py`)
- URL Discovery: `sitemap.xml` + domain filtering
- Extraction: Trafilatura (primary), BeautifulSoup fallback
- Chunking: token-aware (tiktoken), `chunk_size=1000`, `chunk_overlap=200`
- Embeddings: batched calls to OpenAI embeddings
- Storage: upsert chunks + metadata to Pinecone (via `src/storage/pine.py`)

### 6) Storage Layer (`src/storage/`)

#### a) Supabase DB (`db.py`)
- Table: `chat_logs` with columns for query, is_oos, retrieved ids/urls, rerank scores, answer, citations, model, latency, cost, flags
- Views + indexes for analytics and performance

#### b) Pinecone Vector Store (`pine.py`)
- Index: `zibtek-chatbot-index`, 1536-d cosine (serverless, AWS us-east-1)
- Helpers: create/connect index, batch upsert, stats

### 7) Caching (`src/utils/cache.py`)
- Redis caches with TTLs
  - Retrieval cache: query → documents (10m)
  - Answer cache: query + docIDs + promptHash → answer (60m)
  - BM25 index cache: serialized index (24h)
- Robust initialization and graceful fallbacks

### 8) Evaluation (`src/eval/eval.py`)
- Loads CSV dataset: `question, expected_contains, in_scope`
- Calls `/chat` for each question
- Metrics: scope accuracy, grounded hit@1, latency stats, estimated cost
- Outputs JSON report (summary + per-question details)

## End-to-End Request Flow

1. UI sends POST to `/chat` or `/chat/stream` with `{question, session_id}`
2. API validates + rate limits; checks Redis caches
3. Guardrails run: keyword filter → semantic scope
4. If out-of-scope: return friendly OOS message
5. If in-scope: Hybrid retrieval (vector + BM25) → RRF → optional Cohere rerank
6. Construct prompt with system instructions + top documents + question
7. Generate with OpenAI; extract citations from context
8. Log to Supabase; cache final answer; return response/stream

## Technology Stack

- UI: Streamlit
- API: FastAPI, Pydantic, SlowAPI
- LLM: OpenAI GPT-4o-mini
- Embeddings: OpenAI `text-embedding-3-small`
- Vector DB: Pinecone (serverless, cosine)
- Keyword Search: rank-bm25 (BM25Okapi)
- Rerank: Cohere Rerank v3.5
- Cache: Redis
- DB: Supabase (PostgreSQL)
- Scraping: Trafilatura, BeautifulSoup
- Tokens: tiktoken
- Retries: tenacity
- Env: python-dotenv

## Key Metrics (from example eval run)

- Overall scope accuracy: ~100% (on sample set)
- Grounded hit@1: ~70% (in-scope)
- Avg request latency: ~9.5s; median ~12.6s (end-to-end)
- Cache hits: sub-100ms typical

These metrics depend on network conditions, dataset size, and configuration (rerank on/off, cache warmup, etc.).

## Deployment

- Local: `python run_demo.py` (starts FastAPI + Streamlit)
- Procfile:
  - `web`: `uvicorn src.app.server:app --host 0.0.0.0 --port $PORT`
  - `ui`: `streamlit run src/ui/app.py --server.port $PORT --server.address 0.0.0.0`
- Docker: `infra/docker` contains Dockerfile/compose (and Redis setup)

## Environment Variables

- Required:
  - `OPENAI_API_KEY`
  - `PINECONE_API_KEY`
  - `SUPABASE_URL`, `SUPABASE_ANON_KEY`
  - `COHERE_API_KEY` (for reranking if enabled)
- Optional / Defaults:
  - `DATASET_DOMAIN` (e.g., https://www.zibtek.com)
  - `RERANK_ENABLED`, `CACHE_ENABLED`, `RATELIMIT_ENABLED`
  - `REDIS_URL` (default redis://localhost:6379/0)
  - `STREAMLIT_AUTH_SHARED_SECRET`
  - `MIN_SCOPE_SIM` (semantic scope threshold)

## Repository Structure (key files)

```
src/
├── app/            # FastAPI server
├── guards/         # Guardrails
├── retrieval/      # Retriever, hybrid, rerank
├── ingest/         # Data pipeline
├── storage/        # Supabase + Pinecone helpers
├── ui/             # Streamlit UI
├── eval/           # Evaluation tools
└── utils/          # Cache, retries, db init
```

## Notes & Best Practices

- Start with cache + rerank disabled to validate the happy path, then enable gradually
- Keep BM25 index cached in Redis to speed up hybrid retrieval startup
- Monitor Supabase logs to ensure all chat events are persisted
- Keep embeddings model and Pinecone dimension in sync (1536 for `text-embedding-3-small`)
- Respect external service limits; retry with backoff on failures
