# BlightSanest: Project Overview & Architecture

**An anagram of "Stable Insights"**

---

## 1. Executive Summary

**BlightSanest** is a domain-agnostic smart journaling platform that enables users to store, organize, and interact with their personal data across any domain through conversational AI. It is neither a diagnostic nor predictive tool—it is a platform for intelligently managing and extracting insights from private data.

The project serves dual purposes:
- **Portfolio piece**: A production-grade, fully architected system demonstrating enterprise-level design decisions
- **Learning vehicle**: An AWS certification learning project that integrates real AWS services end-to-end

---

## 2. Background & Lineage

BlightSanest unifies two predecessor projects:

### 2.1 Original BlightSanest (PubSub CLI)
- RabbitMQ-based publish-subscribe CLI application
- Specialized in cryptocurrency analysis
- Analyzed third-party API data and persisted results to a database
- Proved the concept of async data processing and real-time sharing

### 2.2 BlightSanest Health (Health-Focused RAG API)
- ExpressJS + TypeScript backend
- Structured medical data via openEHR standards
- Sentence Transformer embeddings + Google Gemini LLM
- Users could query their medical records conversationally
- Demonstrated RAG viability for private, domain-specific data

### 2.3 BlightSanest Unified Platform
The new project combines insights from both predecessors into a single, generalized platform that is:
- **Domain-agnostic** (not locked to health)
- **Modular** (domain packages are pluggable)
- **Scalable** (AWS infrastructure for millions of users)
- **Privacy-first** (per-user indexes, all data private by default v1)

---

## 3. Vision & Goals

### 3.1 Core Vision
A unified platform where users:
- **Own** their private data across any domain
- **Query** it conversationally through RAG
- **Search** it directly when they prefer specificity
- **Share** it opt-in with others (v2 community layer)
- **Extend** it with domain-specific modules

### 3.2 Supported Domains
Health, finance, music, games, fitness, productivity, and more. Each domain gets a modular service layer within the API.

### 3.3 Privacy Model
- **Version 1**: All data is private by default. No sharing.
- **Version 2**: Opt-in community intelligence layer with anonymization and explicit consent.

---

## 4. System Architecture

### 4.1 Four Core Components

The system is built around **four interdependent legs** with the **API at the center**:

```
┌────────────────────────────────────────────┐
│                 API (Go)                   │
│         Central Orchestrator               │
│   • Domain modules (health, finance, etc)  │
│   • Request routing                        │
│   • Authentication & authorization         │
└─────────┬──────────────────────┬───────────┘
          │                      │
          ├──────────────────────┼──────────────────┐
          │                      │                  │
    ┌─────▼──────┐        ┌─────▼──────┐    ┌─────▼──────┐
    │  Database  │        │   RAG      │    │  PubSub    │
    │ (Aurora +  │        │ (Python)   │    │   (Go)     │
    │ pgvector)  │        │            │    │            │
    └────────────┘        └────────────┘    └────────────┘
```

| Component | Language | Responsibility |
|-----------|----------|-----------------|
| **API** | Go | Central orchestrator. Routes requests to RAG, PubSub, and Database. Houses domain service modules. Handles auth, rate limiting, ingestion. |
| **RAG** | Python | Domain-agnostic retrieval and generation. Builds indexes, performs hybrid search, generates LLM responses. Reads vectors from Database. |
| **PubSub** | Go | Opt-in real-time data sharing. Decoupled from RAG. Enables community data exchange layer (v2). |
| **Database** | PostgreSQL (Aurora) | Persistent relational data (users, documents, metadata) + vector embeddings via pgvector. |

### 4.2 Communication Boundaries

Clear separation of concerns:

| Path | Protocol | Data Flow |
|------|----------|-----------|
| **API ↔ Database** | SQL/ORM | Business data, user records, domain data |
| **API ↔ RAG** | gRPC | Query-time only: API sends query + user context to RAG, receives structured response |
| **RAG ↔ Database** | SQL | Read-only index retrieval: vectors, embeddings, metadata |
| **API ↔ PubSub** | gRPC | When user triggers shared data exchange |
| **PubSub ↔ RAG** | None | Deliberately decoupled; do not communicate directly |

**Note**: gRPC is used for all inter-service communication (API ↔ RAG, API ↔ PubSub).

### 4.3 Indexing Strategy

BlightSanest uses a **pre-built index strategy**—not runtime indexing:

1. **Ingestion Time**: When a user saves new data, the API triggers an index update
2. **Silent Indexing**: New documents are chunked, embedded, and indexed asynchronously
3. **Persistent Storage**: Indexes (BM25 structures + vectors) are saved to S3 and Redis
4. **Query Time**: RAG loads the user's index on-demand, performs hybrid search, generates response
5. **No Rebuilding**: Queries never trigger index rebuilds—only updates trigger reindexing

This approach:
- ✅ Avoids expensive indexing at query time
- ✅ Scales to large document collections
- ✅ Supports per-user privacy (each user has isolated index)
- ✅ Enables incremental updates (append-only)

---

## 5. Technology Stack

### 5.1 Core Services

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Backend / API** | Go | High-performance, concurrent service. Central orchestrator. |
| **RAG Service** | Python 3.11+ | Machine learning, embeddings, search algorithms, LLM integration. |
| **PubSub Service** | Go | Real-time messaging and event distribution. |
| **Service Communication** | gRPC + Protocol Buffers | Type-safe, high-performance RPC between services. |

### 5.2 Data & Storage

| Component | AWS Service | Purpose |
|-----------|------------|---------|
| **Vectors & Metadata** | Amazon Aurora PostgreSQL (Serverless v2) + pgvector extension | Relational data, user accounts, document metadata, vector embeddings. |
| **Large Indexes & Embeddings** | AWS S3 | Source of truth for BM25 structures, chunk embeddings, term frequencies, document mappings. |
| **Hot Index Cache** | AWS ElastiCache (Redis) | In-memory cache for active users' indexes. TTL-based expiration. Optional (S3 fallback). |

### 5.3 ML & LLM

| Component | Technology | Environment |
|-----------|-----------|-------------|
| **Embeddings** | Sentence Transformers (all-MiniLM-L6-v2) | Local dev & production (runs locally or containerized). |
| **LLM (Development)** | Ollama (Llama 3, Mistral, Phi-3) | Local development via Docker. |
| **LLM (Production)** | AWS Bedrock | Managed LLM service; replaces Ollama in production. |
| **Search Algorithm** | Hybrid Search + RRF | Combines BM25 (lexical) and semantic (embedding-based) search via Reciprocal Rank Fusion. |

### 5.4 Security & Access

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Authentication** | AWS Cognito | User identity, password management, MFA (optional). |
| **Authorization** | Custom Middleware (API) | Internal access control, per-user data isolation. |
| **Encryption in Transit** | TLS 1.3 | All API and gRPC communication encrypted. |
| **Secrets Management** | AWS Secrets Manager | API keys, database credentials, LLM credentials. |

### 5.5 Infrastructure & CI/CD

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Containerization** | Docker + Alpine Linux | Lightweight, production-ready images. All services containerized. |
| **Orchestration** | AWS EKS (Kubernetes) | Managed container orchestration. Auto-scaling, self-healing, networking. |
| **Local Development** | Docker Compose | Multi-service local environment (Postgres, Redis, Ollama, API, RAG, PubSub). |
| **CI/CD** | GitHub Actions → AWS | Automated testing, builds, pushes to ECR, deploys to EKS. |
| **Infrastructure** | AWS (IaC) | Serverless-first approach. Pay-on-use pricing model. |

---

## 6. Data Flow & Query Pipeline

### 6.1 Ingestion Flow

```
User Data (file / manual / 3rd-party API)
    ↓
API → Validate & Normalize
    ↓
Store in S3 (original files)
Store metadata in Aurora (document record)
    ↓
RAG Service (triggered silently)
    ├─ Chunk documents
    ├─ Generate embeddings (Sentence Transformers)
    ├─ Build BM25 index
    ├─ Save structures to S3
    └─ Populate pgvector in Aurora
    ↓
Update Redis cache (active user's indexes)
    ↓
✅ Ready for queries
```

### 6.2 Query Flow (RAG Mode)

```
User Query
    ↓
API receives query + user context
    ↓
gRPC call to RAG service
    ├─ Load user's indexes from Redis (or S3 fallback)
    ├─ Hybrid search (BM25 + semantic)
    ├─ Retrieve top-k documents
    └─ Rank via RRF
    ↓
LLM Prompt Construction
    ├─ System prompt (role, rules, purpose)
    ├─ Retrieved documents (context)
    └─ User query
    ↓
Call LLM (Ollama dev / Bedrock prod)
    ↓
Structured Response
    ├─ Status (found / not found)
    ├─ LLM-generated answer
    └─ Source documents + confidence
    ↓
API returns to user
    ↓
✅ Conversational answer, grounded in user's data
```

### 6.3 Query Flow (Search Mode)

```
User Search Query
    ↓
API receives query
    ↓
gRPC call to RAG service
    ↓
Hybrid search (BM25 + semantic)
    ↓
Retrieve matching documents (no LLM)
    ↓
Return structured list of results
    ↓
✅ Direct retrieval, user gets specific entries
```

---

## 7. Storage Architecture

### 7.1 S3 Layout (Source of Truth)

All data organized under per-user directory structure:

```
s3://blightsanest-bucket/
└── users/
    └── {user_id}/
        ├── documents/              # Original document files
        │   ├── doc_1.pdf
        │   └── doc_2.txt
        ├── inverted_index/         # BM25 structures
        │   └── index.msgpack
        ├── docmap/                 # Document ID mappings
        │   └── docmap.msgpack
        ├── term_frequencies/       # Term statistics
        │   └── tf.msgpack
        ├── doc_lengths/            # Document lengths
        │   └── lengths.msgpack
        ├── chunk_embeddings/       # Numpy arrays (binary)
        │   └── embeddings.npy
        └── chunk_metadata/         # Chunk info (JSON)
            └── metadata.json
```

**Key principle**: S3 is the **authoritative store**. All indexes persist here. Local/Redis copies are disposable.

### 7.2 Aurora PostgreSQL Schema

```sql
-- Users table
CREATE TABLE users (
    id UUID PRIMARY KEY,
    username VARCHAR(255) UNIQUE,
    email VARCHAR(255) UNIQUE,
    hashed_password VARCHAR(255),
    created_at TIMESTAMP,
    updated_at TIMESTAMP
);

-- Documents table
CREATE TABLE documents (
    id UUID PRIMARY KEY,
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    filename VARCHAR(255),
    content_type VARCHAR(255),
    size_bytes BIGINT,
    created_at TIMESTAMP,
    updated_at TIMESTAMP
);

-- Vector embeddings (via pgvector extension)
CREATE TABLE embeddings (
    id UUID PRIMARY KEY,
    document_id UUID REFERENCES documents(id) ON DELETE CASCADE,
    chunk_id INT,
    embedding vector(384),  -- all-MiniLM-L6-v2 dimensionality
    metadata JSONB,
    created_at TIMESTAMP
);
```

### 7.3 Redis Cache (Hot Layer)

Active users' indexes cached in Redis with TTL:

```
Key Pattern: users/{user_id}/{index_type}

Examples:
  - users/user_123/inverted_index       → BM25 structures (TTL: 1 hour)
  - users/user_123/chunk_embeddings     → Embeddings array (TTL: 1 hour)
  - users/user_123/chunk_metadata       → Metadata (TTL: 1 hour)
```

**Behavior**:
- ✅ Cache hit → Serve from Redis (milliseconds)
- ❌ Cache miss → Load from S3, update Redis, serve (seconds)
- ❌ Redis down → Load from S3, skip cache (non-fatal)

---

## 8. Query Modes

### 8.1 RAG Mode (Conversational)

User asks a question grounded in their own data:

**Example**: "What were my main health concerns in Q1?"

→ RAG retrieves relevant health records, passes them to LLM, LLM synthesizes an answer.

**Characteristics**:
- Grounded in user's own data
- LLM generates prose response
- Cites sources (document references)
- Best for synthesis, patterns, summaries

### 8.2 Search Mode (Direct Retrieval)

User searches for specific entries without LLM synthesis:

**Example**: "Show me all entries with fever over 100°F"

→ RAG performs hybrid search, returns matching documents with scores.

**Characteristics**:
- No LLM involved
- Fast, deterministic results
- Best for fact-finding, exact matches
- Lower latency, lower cost

---

## 9. Key Design Principles

### 9.1 Privacy & Isolation
- All data is private by default (v1)
- Each user has isolated indexes
- No cross-user data access
- Opt-in sharing only (v2)

### 9.2 Scalability
- Per-user indexing (not global)
- On-demand index loading (not startup caching)
- Serverless AWS infrastructure (auto-scaling)
- Horizontal scaling via Kubernetes

### 9.3 Reliability
- S3 as source of truth (durable, replicated)
- Redis as optional cache (non-critical)
- Automatic S3 fallback on cache miss
- Alembic migrations for schema versioning

### 9.4 Cost Efficiency
- Serverless-first (Aurora Serverless, Lambda-ready)
- Pay-on-use pricing
- Alpine containers (minimal resource overhead)
- Efficient serialization (MessagePack, not JSON)

### 9.5 Developer Experience
- Single monorepo (all code in one place)
- Docker Compose for local development
- Clear separation of concerns (API, RAG, PubSub)
- TypeConverter abstraction for storage (testable, mockable)

---

## 10. Deployment Topology

### 10.1 Local Development
```
Docker Compose:
  ├─ PostgreSQL 15 (Alpine)
  ├─ Redis 7 (Alpine)
  ├─ Ollama (LLM)
  ├─ RAG Service (Python)
  ├─ API Service (Go)
  └─ PubSub Service (Go)

All services on shared Docker network. Moto mocks S3 locally.
```

### 10.2 Production (AWS)
```
AWS Infrastructure:
  ├─ Aurora PostgreSQL (Serverless v2)
  ├─ ElastiCache Redis (cluster mode disabled)
  ├─ S3 bucket (indexes + documents)
  ├─ EKS Cluster (Kubernetes)
  │   ├─ RAG Service deployment (Python)
  │   ├─ API Service deployment (Go)
  │   └─ PubSub Service deployment (Go)
  ├─ AWS Cognito (authentication)
  ├─ AWS Bedrock (LLM)
  ├─ ECR (container registry)
  ├─ CloudWatch (logging & monitoring)
  └─ GitHub Actions (CI/CD)
```

---

## 11. Supported Data Ingestion

### 11.1 Methods
1. **File Upload** → Stored in S3, indexed asynchronously
2. **Manual Entry** → Via API, indexed on save
3. **Third-Party Integration** → Decided per domain
   - Health: openEHR APIs, FHIR endpoints
   - Finance: CoinGecko, Alpha Vantage, Plaid
   - Fitness: Strava, Fitbit, Apple HealthKit
   - etc.

### 11.2 Supported Formats
- Text files (.txt, .md, .csv)
- Documents (.pdf, .docx)
- JSON/JSONL (structured data)
- Domain-specific formats (handled by domain modules)

---

## 12. Security & Compliance

### 12.1 Authentication Flow
1. User registers via Cognito
2. Cognito issues JWT tokens
3. API validates JWT on each request
4. Cognito provides MFA, password reset, session management

### 12.2 Authorization
- Internal API middleware checks user_id in JWT
- All database queries filtered by user_id
- Cross-user access is impossible by design
- Rate limiting per user ID

### 12.3 Data Protection
- All data encrypted at rest (S3, RDS, Redis)
- All communication encrypted in transit (TLS 1.3)
- Secrets (API keys, passwords) stored in AWS Secrets Manager
- Audit logs in CloudWatch for compliance

---

## 13. Roadmap Summary

| Phase | Status | Focus | Est. Duration |
|-------|--------|-------|---------------|
| 1 | ✅ **Complete** | RAG service, indexing, storage layer | Done |
| 2 | 🚀 **In Progress** | Bedrock integration, end-to-end testing | 1-2 weeks |
| 3 | ⏳ **Next** | Go API service, domain modules | 3-4 weeks |
| 4 | ⏳ **Planned** | gRPC integration between API & RAG | 2-3 weeks |
| 5 | ⏳ **Planned** | PubSub service, real-time sharing | 3-4 weeks |
| 6 | ⏳ **Planned** | Infrastructure, EKS, CI/CD, deployment | 4-6 weeks |
| 7 | ⏳ **Future** | Version 2: shared indexes, community layer | TBD |

---

## 14. Key Learnings & Decisions

### Why AWS over alternatives?
- Unified ecosystem (great for certification learning)
- Serverless-first pricing model (cost-efficient)
- Production-ready security (Cognito, Secrets Manager, KMS)
- Strong ML/LLM integrations (Bedrock)

### Why separate RAG, API, and PubSub services?
- Clear separation of concerns
- Independent scaling (RAG can scale separately from API)
- Easier testing (each service has clear contract)
- Future-proof (services can be replaced/upgraded independently)

### Why per-user indexes (not global)?
- Privacy (no cross-user data leakage)
- Simplicity (no complex access control needed)
- Scalability (adds linear cost per user, not exponential)
- v2 can build shared layer on top

### Why MessagePack serialization?
- Binary format (smaller than JSON)
- Language-agnostic (Python, Go, JavaScript support)
- Faster serialization/deserialization
- Handles arbitrary Python types (sets, tuples, Counter, defaultdict)

### Why Alembic for migrations?
- Version control for schema
- Reversible (can rollback)
- Reproducible deployments
- Industry standard for SQLAlchemy projects

---

## 15. For Developers

**Starting a new session?** Use these files:
- `BlightSanest_ProjectBrief.docx` — High-level vision & build order
- `BlightSanest_RAG_AWS_Context.txt` — AWS implementation strategy
- `BlightSanest_RAG_Context.txt` — Current RAG completion status

**Contributing?**
- Keep the four legs (API, RAG, PubSub, Database) loosely coupled
- All inter-service communication through gRPC
- S3 is the source of truth; Redis is optional
- Per-user indexing is a hard constraint (v1)
- Test locally with Docker Compose before AWS

---

**Version**: 1.0  
**Last Updated**: June 2026  
**Status**: RAG service phase (Phase 1) complete. Bedrock integration in progress.
