# BlightSanest: Progress & Roadmap

**Current Status**: Phase 1 (RAG Service) finishing — Phase 3 (Go API) scaffolding started in parallel  
**Last Updated**: September 2026

---

## 1. Executive Status

| Metric | Status | Notes |
|--------|--------|-------|
| **Phase 1: RAG Service** | 🟡 In Progress | Indexing, hybrid search, storage, RAG class, Ollama + Bedrock providers implemented. Bedrock untested; per-component unit tests, storage-layout and credential fixes pending. |
| **Phase 2: Database** | 🟡 Partial | `users` and `documents` tables, SQLAlchemy models, first Alembic migration done. pgvector / `embeddings` table not started. |
| **Phase 3: API Service** | 🟡 Started | Go HTTP server with graceful shutdown, typed env config, slog logger (not yet wired in), domain types. No routes, handlers, services, or repositories yet. |
| **Phase 4: gRPC Integration** | ⏳ Not Started | No `proto/` directory yet. |
| **Phase 5: PubSub Service** | ⏳ Not Started | No `pubsub/` directory yet. |
| **Phase 6: Infrastructure** | ⏳ Not Started | No Dockerfiles, IaC, or CI. Docker Compose runs Postgres, Redis, Ollama only. |
| **Phase 7: Version 2** | ⏳ Future | Shared community layer (post v1 release). |

---

## 2. Phase 1: RAG Service (IN PROGRESS)

### 2.1 What's Complete ✅

#### 2.1.1 Core RAG Logic (`rag/rag/rag.py`)
- ✅ `RAG` class
  - Takes query + retrieved documents
  - Formats documents as `[id] content` for citation in the prompt
  - Calls the injected LLM `generate(user_prompt, system_prompt)` function
  - Parses the JSON reply into a Pydantic `RagResponse` (`status`: found / not found, `response`)
  - Raises `ValueError` on no response, non-JSON, or missing fields
- ✅ Dependency injection: the LLM provider is passed via the constructor; no provider selection inside `RAG`

#### 2.1.2 LLM Providers (`rag/llm/`)
- ✅ `llm_ollama` — async Ollama client, `OLLAMA_HOST` env (default `http://localhost:11434`), default model `gemma3`
- ✅ `llm_bedrock` — Bedrock Converse API (model-agnostic), `AWS_REGION_NAME` / `BEDROCK_MODEL_ID` env (default `anthropic.claude-3-haiku-20240307-v1:0`), boto3 call run via `asyncio.to_thread`
  - Uses the default AWS credential chain (IAM role compatible)
  - Returns `None` on client errors or malformed responses
  - ⚠️ Not covered by any test yet

#### 2.1.3 Inverted Index — BM25 (`rag/inverted_index/`)
- ✅ BM25 from scratch (k1 = 1.5, b = 0.75)
  - Token → document ID mapping, term frequencies, document lengths, docmap
- ✅ `build()` / `save()` / `load()` entry points
- ✅ Persisted via `Storage` as MessagePack blobs: `inverted_index`, `docmap`, `term_frequencies`, `doc_lengths`

#### 2.1.4 Semantic Index (`rag/semantic_index/`)
- ✅ Sentence Transformers `all-MiniLM-L6-v2` (384-dim)
- ✅ Sentence-based chunking with sliding window (`semantic_chunk`: 4 sentences per chunk, 1 sentence overlap)
- ✅ Chunk metadata: `document_id`, `chunk_index`, `total_chunks`
- ✅ Chunks resolved back to documents via the stable `docmap` by `document_id` (no positional indexes)
- ✅ `build_chunk_embeddings()` / `create_or_load_chunk_embeddings()`; persisted as `chunk_embeddings` (numpy) and `chunk_metadata`
- ✅ Search: per-document max cosine similarity across chunks

#### 2.1.5 Hybrid Search — RRF (`rag/search/`)
- ✅ Reciprocal Rank Fusion (k = 60) over the union of BM25 and semantic results
- ✅ Content resolved from semantic results or the BM25 docmap
- ✅ Default search limit 50

#### 2.1.6 Storage Layer (`rag/storage/`)
- ✅ `Storage` class used by both indexes
  - `upload_data(name, data)` → S3 (authoritative), then Redis with TTL (default 3600s)
  - `load_data(name)` → Redis first, falls back to S3
  - Redis errors on upload or load are logged and non-fatal
- ✅ Keys are per-user: `{user_id}/{name}` in both S3 and Redis
- ⚠️ See §2.2.2 for gaps against the target design (`users/` prefix, credentials, cache repopulation)

#### 2.1.7 Type Conversion & Serialization (`rag/type_converter/`)
- ✅ `TypeConverter` with dynamic type registry + MessagePack
  - Handles set, tuple, Counter, OrderedDict, defaultdict, numpy arrays, Pydantic models
  - Recursive processing of nested structures

#### 2.1.8 Testing (`rag/test/`)
- ✅ `test_type_converter.py` — 32 tests
- ✅ `test_storage.py` — 8 tests: model registration, upload success/empty/S3 failure/Redis failure non-fatal, cache hit, cache-miss S3 fallback, S3 failure returns `None`
- ✅ `test_rag.py` — 1 end-to-end test: build → save → reload → RRF search → `RAG` with a mocked `generate`
- ✅ Infrastructure: pytest + pytest-asyncio (`asyncio_mode = strict`), moto `mock_aws`, `conftest.py` fixtures (user, documents, S3 bucket, real Redis)

#### 2.1.9 Local Development Environment
- ✅ Docker Compose: PostgreSQL 15 Alpine, Redis 7 Alpine, Ollama — shared `blightsanest_network`
  - RAG, API, PubSub services present but commented out (no Dockerfiles yet)
- ✅ Python 3.12 (`rag/.python-version`), dependencies in root `requirements.txt`

### 2.2 What Remains for Phase 1 ⏳

#### 2.2.1 Bedrock Validation
- ⏳ Unit tests for `llm_bedrock` with mocked Converse responses (success, client error, malformed shape)
- ⏳ Integration test against real Bedrock (dev account)
- ⏳ Benchmark Ollama vs Bedrock (quality, latency, cost)

#### 2.2.2 Align Storage With the Target Design
- ⏳ Key prefix `users/{user_id}/` (currently `{user_id}/`)
- ⏳ Credentials: `Storage` currently builds its S3 client from per-user `aws_access_key_id` / `aws_secret_access_key` / `region` fields on `User`. Target design is a service IAM role (no per-user secrets); decision pending
- ⏳ Repopulate Redis on S3 fallback (currently a cache miss does not write back)
- ⏳ Redis socket timeouts so an unreachable Redis falls back instead of hanging
- ⏳ Chunk-metadata migration: indexes built before the `document_index` → `document_id` change must be rebuilt or versioned

#### 2.2.3 Pre-Built Index Enforcement
- ⏳ `HybridSearch.__init__` currently calls `create_or_load_chunk_embeddings()` and `InvertedIndex.load()`, which build and save the index if storage is empty. Split this so the query path only loads, and building happens only at ingestion

#### 2.2.4 Test Coverage
- ⏳ Unit tests for `InvertedIndex` (BM25 scoring), `SemanticIndex`, `HybridSearch` (RRF), chunking helpers
- ⏳ Tests for `llm_ollama` / `llm_bedrock`

#### 2.2.5 Tooling & Production Readiness
- ⏳ Move to `uv` + `pyproject.toml` + Ruff as specified in `CLAUDE.md` (no `pyproject.toml` / `uv.lock` yet)
- ⏳ Structured logging (module loggers instead of `botocore.client.logging` / `print`)
- ⏳ gRPC server (`rag/server.py` is currently an empty stub) — see Phase 4
- ⏳ Error handling: missing indexes, S3 timeouts, Bedrock throttling, empty result sets

### 2.3 Phase 1 Completion Criteria

Phase 1 is **COMPLETE** when:
- ✅ Core RAG logic implemented and covered by the e2e test
- ✅ Storage layer (S3 + Redis fallback) working and tested
- ✅ Ollama and Bedrock providers implemented
- ⏳ Bedrock tested (mocked + real)
- ⏳ Storage layout and credentials aligned with the target design
- ⏳ Query path never builds indexes
- ⏳ Per-component unit tests in place
- ⏳ Ready for Phase 3/4 (API integration over gRPC)

---

## 3. Phase 2: Database (PARTIAL)

### 3.1 Completed Work ✅

- ✅ SQLAlchemy ORM (`models/`)
  - `User`: `id` (UUID), `username` (unique), `email` (unique), `hashed_password`, `created_at`, `updated_at`
  - `Document`: `id` (UUID), `user_id` (FK → users), `created_at`, `updated_at`
  - ORM-level cascade (`all, delete-orphan`) from User → Documents
- ✅ Alembic (`migrations/`)
  - First migration `811ecb922478` creates `users` and `documents`; reversible
- ✅ Local PostgreSQL 15 Alpine container via Docker Compose

### 3.2 Remaining ⏳

- ⏳ pgvector extension (the `postgres:15-alpine` image does not include it; needs a pgvector image and a migration)
- ⏳ `embeddings` table (document_id, chunk_id, `vector(384)`, JSONB metadata)
- ⏳ `documents` metadata columns (filename, content_type, size_bytes)
- ⏳ Database-level `ON DELETE CASCADE` on `documents.user_id` (current FK has no ON DELETE clause)
- ⏳ RAG read-only SQL access (no SQL code in `rag/` yet)
- ⏳ Aurora Serverless v2 provisioning (only the local container exists)

---

## 4. Phase 3: API Service (STARTED)

### 4.1 Scope

The API is the central orchestrator. It receives user requests, validates them, routes to RAG/PubSub, handles domain logic, manages ingestion, and coordinates with the database.

### 4.2 What Exists ✅

- ✅ Go module `github.com/mehmetcagriekici/blightsanest_stable_insights/api` (Go 1.26)
- ✅ `cmd/api/main.go` — signal-aware context (SIGINT/SIGTERM), config load, `run()` that handles both shutdown signals and server start failures
- ✅ `cmd/api/server.go` — `http.Server` wrapper with `ServeMux` (no routes yet), `start()` / `kill()` with 5s graceful shutdown
- ✅ `internal/config` — typed `Config` loaded once from env: `PORT` (8080), `CUSTOM_BUFFER_SIZE` (8192), `ENV` (`development`)
- ✅ `cmd/api/logger.go` — `log/slog` JSON logger to a buffered file, debug level in development, `env` + `hostname` attributes
  - ⚠️ Not yet called from `main`
- ✅ `internal/domain` — `RagRequest`, `RagResponse`, `Document` types

### 4.3 Remaining ⏳

- ⏳ Wire the logger (request ID + user ID per request)
- ⏳ Handlers → services → repositories layers
- ⏳ Repositories for users/documents (tests against real Postgres via Docker Compose)
- ⏳ Server timeouts (`ReadHeaderTimeout` etc.)
- ⏳ Document ingestion endpoint (store in S3, trigger index build)
- ⏳ Cognito JWT validation, per-user authorization, rate limiting
- ⏳ Domain modules (health, finance, music, …)
- ⏳ Tests (none yet)
- ⏳ gRPC clients for RAG and PubSub (Phase 4)

### 4.4 Target Architecture

```
User Request
    ↓
API Gateway (HTTP or REST)
    ├─ Cognito Authentication
    ├─ Rate Limiting
    ├─ Request Validation
    └─ Routing
    ↓
Go API Service
    ├─ Domain Modules (health, finance, music, etc.)
    ├─ Document Ingestion Handler
    ├─ User Profile Management
    ├─ gRPC Client → RAG Service
    ├─ gRPC Client → PubSub Service
    └─ SQL queries → Aurora
    ↓
Response → User
```

### 4.5 Key Responsibilities

| Responsibility | Details |
|---|---|
| **Request Routing** | HTTP → gRPC dispatch. Orchestrate multi-step operations. |
| **Authentication** | Validate Cognito JWT tokens. Extract user_id. |
| **Authorization** | Check per-user access controls. Prevent cross-user access. |
| **Document Ingestion** | Accept file uploads, validate, store in S3, trigger RAG index build. |
| **Domain Modules** | Health (openEHR, FHIR), Finance (CoinGecko, Alpha Vantage), Music, Games, etc. |
| **User Management** | Profile CRUD, settings, preferences. |
| **Rate Limiting** | Per-user request quotas. Prevent abuse. |
| **Error Handling** | HTTP status mapping in handlers only. |
| **Logging** | Structured `slog` logs with request ID and user ID. |

### 4.6 Testing Plan

- Table-driven unit tests for config, services, domain logic
- Repository tests against real PostgreSQL (Docker Compose)
- Integration tests (API ↔ RAG)
- End-to-end tests (user journey)

---

## 5. Phase 4: gRPC Integration (NOT STARTED)

### 5.1 Scope

API and RAG are currently separate codebases with no connection: the RAG e2e test drives `HybridSearch` and `RAG` directly in Python. This phase connects them (and later PubSub) over gRPC.

### 5.2 Changes Required

| Service | Change |
|---|---|
| **Proto** | Create `proto/` with message types for queries, responses, errors. `make proto` target. |
| **RAG** | Implement gRPC server in `rag/server.py`; generated code in `rag/gen/`. |
| **API** | gRPC client stub; generated code in `api/internal/gen/`. |

### 5.3 Proto Definition (Draft)

```protobuf
service RAG {
  rpc Query (QueryRequest) returns (QueryResponse);
  rpc RebuildIndex (RebuildIndexRequest) returns (RebuildIndexResponse);
}

message QueryRequest {
  string user_id = 1;
  string query = 2;
  string mode = 3;  // "rag" or "search"
}

message QueryResponse {
  bool success = 1;
  string status = 2;  // "found" or "not_found"
  string answer = 3;
  repeated Document sources = 4;
}
```

### 5.4 Benefits

- Loose coupling (can replace RAG without changing API)
- Independent scaling
- Network resilience (retries, timeouts)
- Type-safe contracts

---

## 6. Phase 5: PubSub Service (NOT STARTED)

### 6.1 Scope

Real-time, opt-in data sharing between users. Foundation for the community layer (v2). PubSub never talks to RAG directly.

### 6.2 Architecture

```
User A (publishes)
    ↓
API triggers share event
    ↓
gRPC → PubSub Service
    ↓
PubSub Service
    ├─ Persists event to database
    ├─ Redis PubSub broadcast
    └─ Subscribers notified in real-time
    ↓
User B (listening)
    ↓
API ingests shared data (index updates go through the API, not PubSub → RAG)
```

### 6.3 Key Responsibilities

| Responsibility | Details |
|---|---|
| **Event Publishing** | Accept event from API. Store in database. Broadcast to subscribers. |
| **Event Subscriptions** | Users declare interest in topics. Match publishers to subscribers. |
| **Real-Time Delivery** | Redis PubSub for low-latency notification. |
| **Opt-In Consent** | Explicit permission for data sharing. Revokable. |
| **Anonymization** | Strip PII before sharing (v2 requirement). |
| **Rate Limiting** | Prevent spam/DoS on pubsub channels. |

---

## 7. Phase 6: Infrastructure & Deployment (NOT STARTED)

### 7.1 Scope

Containerization, CI/CD pipelines, AWS resource provisioning, and production deployment.

### 7.2 Components

| Component | Status |
|---|---|
| **Dockerfiles** (RAG, API, PubSub) | ⏳ Pending |
| **Docker Compose** | 🟡 Partial — Postgres, Redis, Ollama only; app services commented out |
| **ECR** | ⏳ Pending |
| **EKS** | ⏳ Pending |
| **CI/CD (GitHub Actions)** | ⏳ Pending |
| **Secrets Management** | ⏳ Pending |
| **IaC (Terraform or CloudFormation)** | ⏳ Pending |
| **Monitoring (CloudWatch)** | ⏳ Pending |

### 7.3 Deployment Checklist

- ⏳ Dockerfile per service: Alpine base, multi-stage builds, non-root user, health checks
- ⏳ Kubernetes manifests: Deployments, Services, ConfigMaps, Secrets, Ingress
- ⏳ CI/CD: build, lint, test, push to ECR, deploy to EKS (staging → production), automatic rollback
- ⏳ IaC for Aurora, S3, ElastiCache, EKS

---

## 8. Phase 7: Version 2 – Community Intelligence (FUTURE)

### 8.1 Vision

Move from personal-only (v1) to opt-in community sharing (v2).

### 8.2 Features

- 📊 **Shared Indexes**: Users can opt-in to share anonymized data
- 🌍 **Global Insights**: "Of people who tracked health like you, here's what's common…"
- 🤝 **Community Recommendations**: "5 users in your domain found X helpful"
- 🔐 **Anonymization**: Strip PII, aggregate across users
- ✅ **Explicit Consent**: Every user decides what to share
- 🚫 **Easy Opt-Out**: Users can revoke sharing anytime

### 8.3 Not Planned for v1

- No community features in v1
- All v1 data is private

---

## 9. Overall Roadmap Timeline

### 9.1 Milestones

Original plan dates are kept for reference; phases 1 and 2 have run past their planned end dates, and Phase 3 started in parallel with Phase 1.

| Phase | Planned Duration | Planned Window | Status |
|-------|------------------|----------------|--------|
| Phase 1: RAG | 4 weeks | Apr – Jun 2026 | 🟡 In progress |
| Phase 2: Database | 1 week | May 2026 | 🟡 Partial (pgvector pending) |
| Phase 3: API | 3-4 weeks | Jun – Jul 2026 | 🟡 Started (Jul 2026) |
| Phase 4: gRPC | 2-3 weeks | Jul – Aug 2026 | ⏳ Not started |
| Phase 5: PubSub | 3-4 weeks | Aug – Sep 2026 | ⏳ Not started |
| Phase 6: Infra & Deployment | 4-6 weeks | Sep – Oct 2026 | ⏳ Not started |
| **v1 Release** | — | **Oct 2026 (original target)** | Needs re-planning |
| Phase 7: v2 Features | 4-8 weeks | Nov 2026 – Jan 2027 | ⏳ Future |

### 9.2 Critical Path

```
Phase 1 (RAG) ──┐
Phase 2 (DB)  ──┼─→ Phase 3 (API) → Phase 4 (gRPC) → Phase 5 (PubSub) ┐
                                                   └→ Phase 6 (Infra) ─┴→ v1 Release
```

---

## 10. Current Blockers & Next Actions

### 10.1 Open Decisions

1. **pgvector**: implement the `embeddings` table and RAG read path, or keep embeddings in S3/Redis only for v1
2. **S3 credentials**: per-user credentials on `User` (current code) vs. a single service IAM role (target design)

### 10.2 Phase 1 Next Actions

1. Bedrock tests (mocked + real)
2. `users/{user_id}/` key prefix and chunk-metadata migration
3. Remove index building from the `HybridSearch` query path
4. Unit tests for BM25, semantic index, RRF
5. `uv` + `pyproject.toml` + Ruff

### 10.3 Phase 3 Next Actions

1. Wire the slog logger into `main` (stdout or flushed file)
2. Add server timeouts
3. First handler → service → repository slice (e.g. health check, then documents)
4. Tests for `config.Load` and `run()`

---

## 11. Risk Analysis

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Bedrock latency exceeds requirements | Medium | High | Test early, compare with Ollama. May need caching layer. |
| S3 costs exceed budget | Low | Medium | Monitor S3 usage. Compress indexes. |
| gRPC adoption complexity | Low | Medium | Keep contracts small; start with `Query`. |
| Kubernetes deployment complexity | Medium | High | Start with simple manifests. Iterate. |
| Schedule slip (v1 Oct 2026 target) | High | Medium | Re-plan dates after Phase 1 closes. |

---

## 12. Success Metrics

### 12.1 Phase 1 Completion (RAG)

- End-to-end test passes (ingest → index → query → response) — ✅ with mocked LLM
- Bedrock integration working with <1s latency — ⏳
- 100+ documents indexed without memory issues — ⏳
- Hybrid search producing relevant results — ✅ in e2e test
- Redis fallback working (S3 used when Redis is down) — ✅ unit tested
- All tests passing with high coverage — ⏳ (BM25/semantic/RRF lack unit tests)

### 12.2 v1 Release

- All phases 1-6 complete
- Single user successfully ingests, queries, and retrieves data
- API authentication (Cognito) and rate limiting enforced
- EKS deployment and CI/CD automated
- Cost tracking shows <$50/month for single user
- Documentation complete (API, deployment, operations)

---

## 13. Developer Notes

### 13.1 Code Organization (actual)

```
blightsanest_stable_insights/
├── rag/                          # Python RAG service
│   ├── rag/rag.py               # Core RAG class
│   ├── inverted_index/          # BM25
│   ├── semantic_index/          # Embeddings + chunking
│   ├── search/                  # Hybrid search (RRF)
│   ├── storage/                 # S3 + Redis layer
│   ├── type_converter/          # MessagePack TypeConverter
│   ├── llm/                     # ollama.py, bedrock.py
│   ├── custom_types/            # Pydantic models
│   ├── helpers/, constants/
│   ├── server.py                # gRPC server stub (empty)
│   ├── test/
│   └── pytest.ini
├── api/                          # Go API service
│   ├── cmd/api/                 # main, server, logger
│   └── internal/{config,domain}/
├── models/                       # SQLAlchemy models
├── migrations/                   # Alembic
├── docker-compose.yml
├── requirements.txt
└── CLAUDE.md
```

Not yet present: `proto/`, `pubsub/`, `rag/gen/`, `api/internal/gen/`, Dockerfiles, `Makefile`.

### 13.2 Git Workflow

- Feature branches for each phase
- PRs with test requirements
- Merge to main only after review
- Tag each phase completion (v0.1, v0.2, etc.)

### 13.3 Documentation

- `CLAUDE.md` — architecture constraints and dev commands
- `BlightSanest_Overview.md` — architecture and design
- `rag/README.md` — RAG component notes
- Planned: `DEPLOYMENT.md`, OpenAPI docs, gRPC contract docs

---

## 14. Summary

- **Phase 1 (RAG)**: core complete; Bedrock tests, storage alignment, query-path index building, and unit tests remain
- **Phase 2 (Database)**: users/documents schema done; pgvector pending
- **Phase 3 (API)**: server/config/logger scaffolding in place; no endpoints yet
- **Phases 4-6**: not started
- **v1 target**: October 2026 originally; needs re-planning

---

**Version**: 1.1  
**Last Updated**: September 2026  
**Owner**: Call (Developer)  
**Status**: Active Development
