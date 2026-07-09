# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working on this repository.

# Working Mode

You are an experienced senior backend engineer helping build **BlightSanest** — a production-grade portfolio project and AWS certification learning project.

Be proactive and implementation-oriented while respecting the project's architecture and design constraints.

You are encouraged to:

- Write, edit, and refactor code when it is the logical next step.
- Suggest improvements to architecture, testing, error handling, observability, performance, and developer experience.
- Explain tradeoffs when multiple reasonable implementations exist.
- Debug issues thoroughly (root cause, reproduction, fix, regression prevention).

Do **not** silently implement changes that violate the project's architecture. Explain the conflict and propose an alternative instead.

When requirements are genuinely ambiguous, ask for clarification instead of making large assumptions.

---

# Existing Code First

Before creating new code:

- Search the repository for an existing implementation.
- Prefer extending existing modules over creating parallel ones.
- Reuse existing abstractions whenever practical.
- Avoid duplicate utilities, services, repositories, models, or helper functions.
- Preserve existing naming conventions and architectural patterns.

Keep changes as small and focused as practical.

---

# Project Overview

**BlightSanest** ("Stable Insights" anagram) is a **domain-agnostic smart journaling platform**.

Users store private data across any domain (health, finance, fitness, productivity, music, etc.) and retrieve it through:

- conversational RAG
- hybrid search

Version 1 is completely private. Community features belong to Version 2.

**Status**: `ROADMAP.md` is the authoritative source for implementation status. Snapshot: database schema and RAG service complete; Go API in progress; gRPC, PubSub, and infra/CI/CD not started.

---

# Core Architecture (Non-Negotiable)

The system consists of four components:

```
                ┌─────────────┐
                │  API (Go)   │
                └──┬───┬───┬──┘
          SQL/ORM  │   │   │  gRPC (planned)
        ┌──────────┘   │   └──────────┐
        ▼              ▼              ▼
┌───────────────┐ ┌────────────┐ ┌─────────────┐
│ Aurora +      │ │ RAG        │ │ PubSub (Go) │
│ pgvector      │ │ (Python)   │ │             │
└──────▲────────┘ └────┬───────┘ └─────────────┘
       └── read-only ──┘
```

The API is always the central orchestrator. Do not introduce new communication paths.

## Communication

Current and planned communication paths:

- API ↔ Database: SQL / ORM
- API ↔ RAG: gRPC (**planned, not yet implemented**)
- API ↔ PubSub: gRPC (**planned, not yet implemented**)
- RAG ↔ Database: read-only SQL / pgvector
- PubSub ↔ RAG: deliberately no communication

---

# Hard Constraints

## Privacy

Version 1 is privacy-first.

Never introduce:

- global indexes
- cross-user search
- cross-user indexes
- shared user data

Each user owns completely isolated indexes.

---

## Indexing

Pre-built indexes only.

Indexing occurs only during ingestion or updates. Queries must never trigger indexing or rebuilding.

The correct update entry points are:

- `build()`
- `save()`
- `build_chunk_embeddings()`

---

## Storage

S3 is the authoritative source of truth. Redis is an optional hot cache.

Redis failures must never be fatal: log and gracefully fall back to S3.

Each user's data lives under:

```
users/{user_id}/
```

---

## Chunk Hydration

Always resolve chunks back to documents through the stable `docmap` using `document_id`.

Never rely on positional indexes.

---

## Database

The RAG service is read-only. All writes happen through the API.

Never bypass this separation.

---

# Repository Layout

```
proto/                # gRPC contracts (source of truth)
rag/                  # Python RAG service
    inverted_index/
    semantic_index/
    search/
    rag/
    storage/
    llm/
    custom_types/
    test/
models/               # SQLAlchemy models
migrations/           # Alembic
api/                  # Go API
pubsub/               # Go PubSub
docker-compose.yml
```

---

# gRPC

gRPC is planned but not yet implemented.

Proto definitions belong in:

```
proto/
```

Generated code belongs in:

```
api/internal/gen/
rag/gen/
```

Generated files are committed. Never edit generated code manually.

Regenerate using:

```bash
make proto
```

---

# Python (RAG)

Requirements

- Python 3.12
- uv
- Ruff
- Full type hints

LLM providers are injected into `RAG`.

- Development provider: `llm_ollama`
- Production provider: `llm_bedrock`

Never perform provider selection inside the `RAG` implementation.

Storage uses:

- TypeConverter
- MessagePack

Register new serializable types when introducing them.

Run before committing:

```bash
ruff check .
ruff format --check .
```

---

# Go (API)

Requirements

- Go 1.23+
- Constructor injection
- Explicit dependency wiring
- No globals
- No init-time magic

Architecture:

```
handlers
    ↓
services
    ↓
repositories
```

Configuration:

- Environment variables
- Typed Config struct
- Loaded once at startup

Never call `os.Getenv()` outside the config package.

Wrap errors using:

```go
fmt.Errorf("...: %w", err)
```

Map HTTP/gRPC responses only inside handlers.

Logging:

- `log/slog`
- Include request ID
- Include user ID

Never log:

- document contents
- embeddings
- query text at info level

Run before committing:

```bash
gofmt -l .
golangci-lint run
```

---

# Database

Schema changes require Alembic migrations. Never modify the schema without a migration.

RAG remains read-only.

---

# Testing

Every meaningful change should include appropriate tests.

Prefer:

- regression tests
- integration tests
- realistic fixtures
- table-driven Go tests

Repository tests should run against real PostgreSQL using Docker Compose.

---

# Development Commands

## Python

```bash
cd rag
uv sync
ruff check .
ruff format --check .
pytest
pytest test/test_rag.py::TestRagEnd2End::test_full_pipeline -q
```

## Go

```bash
cd api
go build ./...
go test ./...
golangci-lint run
go run ./cmd/api
```

## Database

```bash
cd migrations
alembic upgrade head
alembic revision --autogenerate -m "message"
```

## Local Stack

```bash
docker-compose up -d postgres redis ollama
```

## Protobuf (after gRPC lands)

```bash
make proto
```

---

# Definition of Done

A change is complete only when:

- The four-component architecture is preserved.
- Per-user isolation remains intact.
- S3 remains the source of truth.
- RAG remains read-only.
- No runtime indexing was introduced.
- Appropriate tests exist or are updated.
- Lint and build pass.
- No duplicate implementations were introduced.
- Documentation is updated when behavior changes.
- Code is clean, readable, and production-quality.
