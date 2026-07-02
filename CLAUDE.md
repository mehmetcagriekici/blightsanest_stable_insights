# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Working Mode: Rubber Duck

This is the user's own portfolio piece and AWS certification learning project. The value is in *them* designing and writing it. Default to rubber-duck mode:

- **Don't write or edit implementation code unless explicitly asked to.** "Explicitly asked" means a direct instruction like "implement this," "write the code," "fix this file," or "add X" — not just discussing a problem, exploring an idea, or asking "how would this work?" out loud.
- When asked to check, review, or debug something: identify and explain the issue (what's wrong, why, what would trigger it) and stop there. Don't jump to a fix. Let the user decide whether and how to fix it.
- When asked "how do I do X" or "what's the right approach here": explain the relevant concepts, trade-offs, and a couple of possible directions — don't hand back a finished design or a diff. Ask questions that help the user reach the answer themselves, the way a rubber duck forces a person to articulate their own reasoning out loud.
- If the user asks a narrow factual question (e.g. "what does this function do," "which file has X," build/test commands), just answer it directly — rubber-duck mode is about not doing their design/implementation work for them, not about being unhelpfully cryptic.
- If unsure whether a request counts as "explicit," ask before writing code.

## Project Overview

BlightSanest ("Stable Insights" anagram) is a domain-agnostic smart journaling platform: users store, organize, and query private data (journal entries, health records, finances, etc.) via conversational AI (RAG) or direct hybrid search. Dual purpose: portfolio piece + AWS certification learning project (Aurora, S3, ElastiCache, Cognito, Bedrock, EKS).

Current phase status: Phase 1 (RAG service) and Phase 2 (Aurora schema/migrations) are complete. Phase 3 (Go API) is in progress. gRPC wiring between API/RAG, the PubSub service, and infra/CI/CD are not started yet.

## Architecture

Four components, API at the center:

```
                 API (Go) — central orchestrator
                  /    |    \
            Database  RAG   PubSub
          (Aurora +  (Python) (Go)
          pgvector)
```

| Path | Protocol | Notes |
|---|---|---|
| API ↔ RAG | gRPC (planned) | query-time only: API sends query + user context, RAG returns a structured response |
| API ↔ PubSub | gRPC (planned) | opt-in data sharing, v2 |
| API ↔ Database | SQL/ORM | business data (users, documents) |
| RAG ↔ Database | SQL, read-only | vector/index retrieval |
| PubSub ↔ RAG | none | deliberately decoupled |

**Pre-built index strategy (load-bearing design constraint):** indexes are built at document-write time, not query time. In `rag/inverted_index/inverted_index.py` and `rag/semantic_index/semantic_index.py`, `load()` / `create_or_load_chunk_embeddings()` only build an index the first time nothing is cached for a user (bootstrap); they never rebuild on a mismatch. `build()`+`save()` and `build_chunk_embeddings()` are the entry points a document create/update/delete path should call directly to reindex a user. Query-time staleness-detection (recomputing a fingerprint of the documents on every search) was tried and rejected — it makes every query pay indexing cost, which contradicts this design.

**Chunk hydration should go through the docmap, not positional indices.** `chunk_metadata` entries are keyed by `document_index` (position in the documents list at index-build time), which drifts if documents are added/removed/reordered afterward — a chunk can end up hydrated against the wrong document, or index out of range. Resolving a chunk back to its document by a stable `document_id` through the docmap avoids this.

## Repository Layout

```
rag/            # Python RAG service (Phase 1, complete)
  inverted_index/    # BM25
  semantic_index/    # Sentence Transformer embeddings + chunking
  search/            # HybridSearch: fuses BM25 + semantic via RRF
  rag/               # RAG class: LLM prompt construction + response parsing
  storage/           # S3 (source of truth) + Redis (optional TTL cache) abstraction
  type_converter/    # MessagePack (de)serialization for non-JSON types
  llm/               # bedrock.py (prod) / ollama.py (dev)
  custom_types/      # Pydantic models (RAG-facing) + DB-facing types
  test/              # pytest suite; test_rag.py is the e2e test
models/         # SQLAlchemy ORM (User, Document) - shared base for Alembic
migrations/     # Alembic migrations for the Aurora/Postgres schema
api/            # Go API (Phase 3, in progress) - HTTP server scaffolding, no routes yet
pubsub/         # Go PubSub service (Phase 5, not started - empty)
docker-compose.yml
```

## Development Commands

### RAG service (Python)

Run from the `rag/` directory (pytest rootdir; `pytest.ini` and imports assume this):

```bash
source .venv/bin/activate        # venv lives at repo root
cd rag
python -m pytest test/ -q                                          # full suite
python -m pytest test/test_rag.py::TestRagEnd2End::test_full_pipeline -q  # single test
```

Tests are fully self-contained — S3 is mocked via `moto`, Redis via `unittest.mock` — no Docker services need to be running to run the suite.

`requirements.txt` is incomplete: it doesn't list `alembic`, `pytest-asyncio`, or `moto`, even though tests and migrations depend on them. Check the existing `.venv` before assuming `pip install -r requirements.txt` alone is sufficient for a fresh environment.

### Go API

```bash
cd api
go build ./...
go vet ./...
gofmt -l .      # list files needing formatting
gofmt -w .      # apply formatting
go run ./cmd/api -port 8899 -data ./data
```

No Go tests exist yet.

### Database migrations

```bash
cd migrations
alembic upgrade head
alembic revision --autogenerate -m "message"
```

Default DB URL (`migrations/alembic.ini`): `postgresql://postgres:password@localhost:5432/blightsanest_dev` — matches the `docker-compose.yml` postgres service credentials.

### Local infrastructure

```bash
docker-compose up -d postgres redis ollama
```

The `rag`, `api`, and `pubsub` services in `docker-compose.yml` are commented out (no Dockerfiles yet) — only Postgres, Redis, and Ollama currently start.

## Key Implementation Notes

- **LLM provider is dependency-injected**: `RAG.__init__` takes a `generate: Callable[[str, str], Awaitable[str]]`. Wire `rag/llm/bedrock.py:llm_bedrock` in production and `rag/llm/ollama.py:llm_ollama` in local dev — don't branch on environment inside the `RAG` class itself.
- **MessagePack type conversion** (`rag/type_converter/type_converter.py`): handles types msgpack/JSON can't natively — `set`, `tuple`, `OrderedDict`, `Counter`, `defaultdict`, `numpy.ndarray`, and registered Pydantic models — by wrapping them as `{"__blightsanest_type__": name, "value": ...}`. Any new non-primitive type stored via `Storage` must be registered with `register_types` (or `register_pydantic_models` for Pydantic models) or it will round-trip as a plain dict/list instead of its original type.
- **Storage fallback behavior** (`rag/storage/storage.py`): S3 is authoritative; Redis is an optional cache. A Redis failure on write or read is logged and ignored — it must never fail an upload or force-error a read that S3 could still serve.
- **Go API server is scaffolding only** (`api/cmd/api/server.go`): `NewServer()` takes no arguments yet (hardcodes port 8899 and ignores CLI flags), and `main.go` has a `run()` loop that hasn't wired the started server's error path into shutdown yet. No routes are registered.
