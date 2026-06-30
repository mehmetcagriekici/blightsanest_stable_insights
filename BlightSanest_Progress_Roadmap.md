# BlightSanest: Progress & Roadmap

**Current Status**: Phase 1 (RAG Service) — Final Testing & Bedrock Integration  
**Last Updated**: June 2026

---

## 1. Executive Status

| Metric | Status | Notes |
|--------|--------|-------|
| **Phase 1: RAG Service** | 🟡 90% Complete | Core indexing/search done. Bedrock integration in progress. |
| **Phase 2: Database** | ✅ 100% Complete | Aurora schema, Alembic migrations, ORM models all working. |
| **Phase 3: API Service** | ⏳ 0% (Blocked) | Awaiting Phase 1 completion. |
| **Phase 4: PubSub Service** | ⏳ 0% (Blocked) | Deferred until API is operational. |
| **Phase 5: Infrastructure** | ⏳ 0% (Blocked) | Dockerfiles and EKS setup pending. |
| **Phase 6: Version 2** | ⏳ 0% (Future) | Shared community layer (post v1 release). |

**Overall Project Progress**: ~25% (1.5 of 6 major phases complete)

---

## 2. Phase 1: RAG Service (IN PROGRESS)

### 2.1 What's Complete ✅

#### 2.1.1 Core RAG Logic
- ✅ `RAG` class fully implemented
  - Takes query + retrieved documents
  - Calls LLM (Ollama currently, Bedrock coming)
  - Returns structured response (Pydantic `RagResponse`)
  - Status: found/not found, LLM answer, source docs
- ✅ Proper dependency injection (LLM `generate` function passed via constructor)
  - Enables clean mocking in tests
  - Production and test code share same implementation

#### 2.1.2 Inverted Index (BM25)
- ✅ BM25 implementation from scratch
  - Token-to-document-ID mapping
  - Term frequency statistics
  - Document length normalization
  - Efficient scoring algorithm
- ✅ Serialization via TypeConverter + MessagePack
  - Handles arbitrary Python types (Counter, OrderedDict, sets)
  - Compact binary format
  - Fully tested (20+ edge cases)
- ✅ Persistence to S3
  - All BM25 structures saved as MessagePack blobs
  - Organized under `s3://bucket/users/{user_id}/`
  - S3 is source of truth

#### 2.1.3 Semantic Index (Embeddings)
- ✅ Sentence Transformer integration (all-MiniLM-L6-v2)
  - 384-dimensional embeddings
  - Efficient chunking (recursive, overlap-aware)
  - Batch processing for speed
- ✅ Chunk metadata tracking
  - Document ID, chunk ID, token counts
  - Source location (for citation)
  - JSON-based storage
- ✅ pgvector integration (Aurora PostgreSQL)
  - Embeddings stored as vectors in database
  - Similarity search via pgvector functions
  - Metadata stored in JSONB columns

#### 2.1.4 Hybrid Search (RRF)
- ✅ Reciprocal Rank Fusion combining BM25 + semantic
  - BM25 captures lexical matching (exact terms)
  - Semantic search captures meaning (synonyms, concepts)
  - RRF avoids bias toward either method
  - Produces ranked document list with scores
- ✅ Tested with real documents and real queries

#### 2.1.5 Storage Layer (S3 + Redis + Aurora)
- ✅ `Storage` abstraction class
  - `upload()` → S3 (authoritative)
  - `get()` → tries Redis first, falls back to S3
  - S3-first failure handling (Redis is optional)
  - Used by both InvertedIndex and SemanticIndex
- ✅ S3 organization (per-user namespace)
  - Indexes isolated under `users/{user_id}/`
  - Prevents cross-user data access
  - Scales to millions of users
- ✅ Redis caching strategy
  - Hot cache for active users
  - TTL-based expiration
  - Automatic fallback to S3
  - Non-critical (system works without it)
- ✅ Aurora pgvector integration
  - Embeddings vectors stored in DB
  - Metadata in JSONB columns
  - Queryable via SQL + pgvector functions

#### 2.1.6 Type Conversion & Serialization
- ✅ `TypeConverter` class (independent layer)
  - Dynamic type registry (extensible)
  - MessagePack serialization
  - Handles: set, tuple, Counter, OrderedDict, defaultdict, numpy arrays, Pydantic models
  - Recursive processing of nested structures
  - 200+ test cases, all passing
- ✅ Language-agnostic format (MessagePack)
  - Can be read/written by Python, Go, JavaScript, etc.
  - Smaller than JSON (important for S3/Redis)
  - Deterministic serialization

#### 2.1.7 Database Schema (Alembic Migrations)
- ✅ Alembic initialization
- ✅ SQLAlchemy ORM models
  - `User` table (id, username, email, hashed_password, timestamps)
  - `Document` table (id, user_id, timestamps)
  - Cascade delete (document deletion removes related data)
  - Relationships defined and tested
- ✅ First migration applied
  - Schema live in Aurora Serverless v2
  - Reversible (can rollback)
  - Version-controlled

#### 2.1.8 Testing
- ✅ Unit tests for all core classes
  - TypeConverter: 200+ test cases
  - Storage: S3/Redis/fallback scenarios
  - InvertedIndex: BM25 scoring
  - SemanticIndex: embedding generation
  - HybridSearch: RRF ranking
- ✅ Integration tests
  - End-to-end: ingest → index → save → load → query → response
  - Real documents, real LLM calls
  - All tested locally with Ollama
- ✅ Test infrastructure
  - pytest with async support
  - moto for S3 mocking (mock_aws, not mock_s3)
  - conftest.py for fixtures
  - pytest.ini configuration (asyncio_mode = auto)
  - Debugging notes documented

#### 2.1.9 Local Development Environment
- ✅ Docker Compose setup
  - PostgreSQL 15 Alpine (2-3 MB per image)
  - Redis 7 Alpine
  - Ollama service (optional, can run separately)
  - All services on shared network
  - Lightweight, fast startup
- ✅ Virtual environment management
  - Single venv at project root
  - requirements.txt (pipreqs-generated)
  - Auto-import support (LunarVim)
  - Documented setup

#### 2.1.10 Documentation
- ✅ Project brief (high-level)
- ✅ AWS context file (strategy)
- ✅ RAG context file (status & roadmap)
- ✅ Code comments (docstrings in all classes)

### 2.2 What Remains for Phase 1 ⏳

#### 2.2.1 Bedrock Integration (1-2 weeks)
- ⏳ Replace Ollama with AWS Bedrock
  - Bedrock is managed LLM service (no local container needed)
  - Supports Claude, Llama, Mistral, etc.
  - IAM-based authentication
  - Production pricing model
- ⏳ Update RAG class
  - Modify `generate()` function to call Bedrock instead of Ollama
  - Handle Bedrock request/response format
  - Error handling for Bedrock timeouts/quota
- ⏳ Credentials management
  - AWS SDK integration (boto3)
  - Secrets Manager for API keys
  - IAM role for EKS pod (no hardcoded credentials)
- ⏳ Testing
  - Mock Bedrock responses for unit tests
  - Test with real Bedrock in integration tests
  - Verify cost tracking (Bedrock charges per token)

#### 2.2.2 End-to-End Manual Script (1 week)
- ⏳ Create `e2e_test.py` or manual script
  - Ingest sample documents (health, finance, etc.)
  - Build indexes
  - Save to S3 and Redis
  - Load indexes back
  - Run queries via RAG
  - Verify response quality
- ⏳ Document learnings
  - Performance metrics (index build time, query latency)
  - Storage usage (S3 size, Redis footprint)
  - LLM quality (Ollama vs Bedrock comparison)
  - Edge cases discovered

#### 2.2.3 Production Readiness Checklist
- ⏳ Error handling
  - Graceful handling of missing indexes
  - S3 timeout recovery
  - Bedrock rate limiting
  - Empty result sets
- ⏳ Logging & observability
  - Structured logging (JSON)
  - Log levels (DEBUG, INFO, WARN, ERROR)
  - Request tracing (for debugging)
  - Performance metrics (latency, throughput)
- ⏳ Caching strategy validation
  - Verify Redis TTL behavior
  - Test cache invalidation (on index updates)
  - Measure cache hit rate
  - Monitor Redis memory usage

### 2.3 Phase 1 Completion Criteria

Phase 1 is **COMPLETE** when:
- ✅ All core RAG logic implemented and tested (DONE)
- ✅ Storage layer working (S3, Redis, Aurora) (DONE)
- ✅ Database schema in place (DONE)
- ⏳ Bedrock integration working (IN PROGRESS)
- ⏳ End-to-end test passes (PENDING)
- ⏳ Production readiness checklist complete (PENDING)
- ⏳ Ready for Phase 3 (API service) to integrate

---

## 3. Phase 2: Database (COMPLETE) ✅

### 3.1 Completed Work

- ✅ Aurora PostgreSQL Serverless v2 schema
  - High availability (automatic failover)
  - Automatic scaling (no capacity planning)
  - Pay-per-second pricing
  - pgvector extension for embeddings
- ✅ SQLAlchemy ORM
  - User model
  - Document model
  - Cascade deletes (proper cleanup)
  - Timestamp tracking (created_at, updated_at)
- ✅ Alembic migration framework
  - Version control for schema changes
  - Reversible migrations (dev-friendly)
  - Production-safe rollback
  - First migration deployed
- ✅ Local PostgreSQL container
  - Docker Compose integration
  - Alpine base image (minimal)
  - Shared Docker network with other services
  - Automatic schema initialization

### 3.2 Not Required for Phase 2

- Embedding storage queries are in pgvector (handled by RAG)
- Search queries are read-only (no custom business queries yet)
- Scaling is handled by Aurora Serverless (auto)

---

## 4. Phase 3: API Service (NOT STARTED)

### 4.1 Scope

The API is the central orchestrator. It receives user requests, validates them, routes to RAG/PubSub, handles domain logic, manages ingestion, and coordinates with the database.

### 4.2 Architecture

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

### 4.3 Key Responsibilities

| Responsibility | Details |
|---|---|
| **Request Routing** | HTTP → gRPC dispatch. Orchestrate multi-step operations. |
| **Authentication** | Validate Cognito JWT tokens. Extract user_id. |
| **Authorization** | Check per-user access controls. Prevent cross-user access. |
| **Document Ingestion** | Accept file uploads, validate, store in S3, trigger RAG reindex. |
| **Domain Modules** | Health (openEHR, FHIR), Finance (CoinGecko, Alpha Vantage), Music, Games, etc. |
| **User Management** | Profile CRUD, settings, preferences. |
| **Rate Limiting** | Per-user request quotas. Prevent abuse. |
| **Error Handling** | Proper HTTP status codes. Informative error messages. |
| **Logging** | Structured logs for debugging and compliance. |

### 4.4 gRPC Contracts (Pending)

Will define `.proto` files for:
- RAG service (query, index rebuild)
- PubSub service (publish event, subscribe)

### 4.5 Estimated Effort

- **Core API**: 2-3 weeks (basic routing, auth, CRUD)
- **Domain modules**: 2-3 weeks (per domain, starting with health)
- **Integration**: 1-2 weeks (gluing services together)
- **Total**: **3-4 weeks**

### 4.6 Testing

- Unit tests for domain logic
- Integration tests (API ↔ RAG)
- End-to-end tests (user journey)
- Load tests (rate limiting, concurrent users)

---

## 5. Phase 4: gRPC Integration (NOT STARTED)

### 5.1 Scope

Currently, API and RAG are in-process (single binary). After Phase 3, they'll be separate services communicating via gRPC.

### 5.2 Changes Required

| Service | Change |
|---|---|
| **RAG** | Add gRPC server. Define `.proto` for Query RPC. |
| **API** | Add gRPC client stub. Call RAG via RPC instead of in-process. |
| **Proto** | Define message types for queries, responses, errors. |

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

### 5.4 Estimated Effort

- **Proto definitions**: 1 week
- **gRPC server (RAG)**: 1 week
- **gRPC client (API)**: 1 week
- **Testing**: 1 week
- **Total**: **2-3 weeks**

### 5.5 Benefits

- ✅ Loose coupling (can replace RAG without changing API)
- ✅ Independent scaling (RAG can scale independently)
- ✅ Network resilience (gRPC retries, timeouts)
- ✅ Type-safe contracts (proto ensures compatibility)

---

## 6. Phase 5: PubSub Service (NOT STARTED)

### 6.1 Scope

Real-time, opt-in data sharing between users. Enables community intelligence layer (v2).

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
    ├─ Receives event
    ├─ Updates local indexes
    └─ RAG incorporates shared data
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

### 6.4 Estimated Effort

- **Core PubSub**: 2-3 weeks
- **Redis integration**: 1 week
- **Testing**: 1-2 weeks
- **Total**: **3-4 weeks**

---

## 7. Phase 5: Infrastructure & Deployment (NOT STARTED)

### 7.1 Scope

Containerization, CI/CD pipelines, AWS resource provisioning, and production deployment.

### 7.2 Components

| Component | Status | Effort |
|---|---|---|
| **Dockerfiles** | ⏳ Pending | 1 week (RAG, API, PubSub) |
| **Docker Compose** | 🟡 Partial | 1 week (integrate all services) |
| **ECR** | ⏳ Pending | 1 week (registry, push pipelines) |
| **EKS** | ⏳ Pending | 2-3 weeks (cluster setup, manifests, networking) |
| **CI/CD Pipelines** | ⏳ Pending | 2-3 weeks (GitHub Actions) |
| **Secrets Management** | ⏳ Pending | 1 week (Secrets Manager integration) |
| **Monitoring** | ⏳ Pending | 1-2 weeks (CloudWatch, alarms) |
| **Total** | **4-6 weeks** |

### 7.3 Deployment Checklist

- ⏳ Dockerfile for each service (RAG, API, PubSub)
  - Alpine base images (minimal)
  - Multi-stage builds (optimize layer caching)
  - Non-root user (security)
  - Health checks
- ⏳ Kubernetes manifests
  - Deployments for each service
  - Services (internal networking)
  - ConfigMaps (configuration)
  - Secrets (credentials)
  - Ingress (external traffic)
  - StatefulSets for databases (if needed)
- ⏳ CI/CD pipeline
  - Build on every commit
  - Run tests
  - Push to ECR
  - Deploy to EKS (staging, then production)
  - Automatic rollback on failure
- ⏳ Infrastructure-as-Code (IaC)
  - Terraform or CloudFormation
  - Define Aurora cluster, S3 bucket, Redis, EKS cluster
  - Reproducible deployments
  - Versioning

### 7.4 Estimated Effort

**Total**: **4-6 weeks**

---

## 8. Phase 6: Version 2 – Community Intelligence (FUTURE)

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
- v2 will be opt-in only

### 8.4 Estimated Effort

**4-8 weeks** (post v1 release)

---

## 9. Overall Roadmap Timeline

### 9.1 Projected Milestones

| Phase | Duration | Start | End | Status |
|-------|----------|-------|-----|--------|
| Phase 1: RAG | 4 weeks | Apr 2026 | Jun 2026 | 🟡 In progress (Bedrock TBD) |
| Phase 2: Database | 1 week | May 2026 | May 2026 | ✅ Complete |
| Phase 3: API | 3-4 weeks | Jun 2026 | Jul 2026 | ⏳ Scheduled |
| Phase 4: gRPC | 2-3 weeks | Jul 2026 | Aug 2026 | ⏳ Scheduled |
| Phase 5: PubSub | 3-4 weeks | Aug 2026 | Sep 2026 | ⏳ Scheduled |
| Phase 6: Infra & Deployment | 4-6 weeks | Sep 2026 | Oct 2026 | ⏳ Scheduled |
| **v1 Release** | — | — | **Oct 2026** | 🎯 Target |
| Phase 7: v2 Features | 4-8 weeks | Nov 2026 | Jan 2027 | ⏳ Future |

### 9.2 Critical Path

```
Phase 1 (RAG) ← Blocker for everything else
    ↓
Phase 2 (Database)
    ↓
Phase 3 (API)
    ↓
Phase 4 (gRPC)
    ↓
Phase 5 (PubSub) ← Can run parallel with Infra
    ↓
Phase 6 (Infra & Deployment)
    ↓
v1 Release → Oct 2026
```

---

## 10. Current Blockers & Next Actions

### 10.1 Phase 1 Completion (Blocking Everything)

**Blocker**: Bedrock integration not yet complete

**Actions**:
1. ✅ Study AWS Bedrock API (Claude, Llama models)
2. ✅ Update RAG class to accept Bedrock as LLM provider
3. ✅ Test with mock Bedrock responses
4. ✅ Test with real Bedrock (dev environment)
5. ✅ Benchmark: Ollama vs Bedrock (quality, latency, cost)
6. ✅ Document learnings
7. ✅ Run end-to-end test with Bedrock
8. → **Unblock Phase 3**

**Timeline**: 1-2 weeks

### 10.2 Phase 3 Readiness (API Service)

**Prerequisites** (blocked until Phase 1 complete):
- ✅ Phase 1 RAG fully functional with Bedrock
- ✅ gRPC proto contracts defined (for Phase 4)

**Preparation** (can start now):
- Define domain module interfaces (health, finance, music, etc.)
- Plan database queries for user/document CRUD
- Design API endpoints (REST, GraphQL, or RPC)
- Plan authentication flow (Cognito integration)
- Outline rate limiting strategy

---

## 11. Risk Analysis

### 11.1 High-Risk Items

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Bedrock latency exceeds requirements | Medium | High | Test early, compare with Ollama. May need caching layer. |
| S3 costs exceed budget | Low | Medium | Monitor S3 usage. Compress indexes. Consider Glacier for archives. |
| gRPC adoption complexity | Low | Medium | Defer gRPC to Phase 4. Keep in-process until necessary. |
| Kubernetes deployment complexity | Medium | High | Start with simple manifest. Iterate. Use AWS best practices. |

### 11.2 Mitigation Strategies

- ✅ Comprehensive testing at each phase
- ✅ Regular cost monitoring (CloudWatch)
- ✅ Phased rollout (dev → staging → production)
- ✅ Automated rollback on failure
- ✅ Regular architecture reviews

---

## 12. Success Metrics

### 12.1 Phase 1 Completion (RAG)

- ✅ End-to-end test passes (ingest → index → query → response)
- ✅ Bedrock integration working with <1s latency
- ✅ 100+ documents indexed without memory issues
- ✅ Hybrid search (BM25 + semantic) producing relevant results
- ✅ Zero data loss (S3 persistence verified)
- ✅ Redis fallback working (S3 used when Redis is down)
- ✅ All tests passing (>95% coverage)

### 12.2 v1 Release (Oct 2026)

- ✅ All phases 1-6 complete
- ✅ Single user successfully ingests, queries, and retrieves data
- ✅ API authentication working (Cognito)
- ✅ API rate limiting enforced
- ✅ EKS deployment successful
- ✅ CI/CD pipeline automated
- ✅ Cost tracking shows <$50/month for single user
- ✅ Documentation complete (API, deployment, operations)
- ✅ Zero critical bugs
- ✅ Production ready (SLA-compliant, resilient)

---

## 13. Developer Notes

### 13.1 Context Switching

Each phase should have its own dedicated chat session with:
- **Phase Brief** (high-level goals)
- **Architecture diagram** (boxes, arrows, communication)
- **Success criteria** (clear completion definition)
- **Testing strategy** (how to validate)

### 13.2 Code Organization

```
blightsanest/
├── rag/                          # Python RAG service
│   ├── src/
│   │   ├── rag.py               # Core RAG class
│   │   ├── inverted_index.py    # BM25
│   │   ├── semantic_index.py    # Embeddings
│   │   ├── hybrid_search.py     # RRF
│   │   ├── storage.py           # S3 + Redis layer
│   │   └── models/              # Pydantic models
│   ├── tests/
│   ├── requirements.txt
│   └── Dockerfile
├── api/                          # Go API service (Phase 3)
├── pubsub/                       # Go PubSub service (Phase 5)
├── migrations/                   # Alembic migrations
├── docker-compose.yml            # Local dev environment
└── README.md
```

### 13.3 Git Workflow

- Feature branches for each phase
- PRs with test requirements
- Merge to main only after review
- Tag each phase completion (v0.1, v0.2, etc.)

### 13.4 Documentation

- Keep README.md updated with latest architecture
- Add DEPLOYMENT.md for infrastructure setup
- Maintain API documentation (OpenAPI/Swagger)
- Document each service's gRPC contracts

---

## 14. Appendix: Key Files & Locations

### 14.1 Project Context Files

- `BlightSanest_ProjectBrief.docx` → High-level vision, build order
- `BlightSanest_RAG_AWS_Context.txt` → AWS strategy, Phase 1 focus
- `BlightSanest_RAG_Context.txt` → Current RAG status, next actions
- `BlightSanest_Overview.md` → Detailed architecture (this file set)

### 14.2 Source Code

- `rag/src/rag.py` → Core RAG class
- `rag/src/inverted_index.py` → BM25 indexing
- `rag/src/semantic_index.py` → Embedding-based indexing
- `rag/src/hybrid_search.py` → RRF fusion
- `rag/src/storage.py` → S3 + Redis abstraction
- `migrations/versions/` → Alembic schema migrations

### 14.3 Configuration

- `docker-compose.yml` → Local development setup
- `pytest.ini` → Test runner configuration
- `requirements.txt` → Python dependencies

### 14.4 Tests

- `rag/tests/test_type_converter.py` → 200+ type conversion tests
- `rag/tests/test_storage.py` → S3/Redis tests
- `rag/tests/test_rag.py` → End-to-end RAG tests

---

## 15. Summary

**BlightSanest** is a ~6-month, multi-phase project to build a production-grade, AWS-certified smart journaling platform.

- **Current Status**: Phase 1 RAG service (90% complete, Bedrock integration pending)
- **Next Phase**: Phase 3 API Service (blocks Phase 4 gRPC)
- **v1 Target**: October 2026
- **Key Principle**: Finish each phase before moving to the next (avoid "wormhole" iteration)

---

**Version**: 1.0  
**Last Updated**: June 2026  
**Owner**: Call (Developer)  
**Status**: Active Development
