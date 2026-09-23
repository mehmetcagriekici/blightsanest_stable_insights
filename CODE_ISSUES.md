# BlightSanest: Known Code Issues

**Last Updated**: September 2026
**Scope**: `api/` (Go), `rag/` (Python), `models/` + `migrations/`, `docker-compose.yml`

This file records problems found in code review. It lists issues only, with no fixes applied. Severity:
- **High**: incorrect behavior, data loss, or security risk
- **Medium**: violates a project constraint in `CLAUDE.md` or a production-readiness gap
- **Low**: quality, style, or tooling

---

## 1. RAG Service (Python)

### 1.1 High

| # | Location | Issue | Impact |
|---|----------|-------|--------|
| R1 | `rag/semantic_index/semantic_index.py:95-102`, `:133` | Chunk metadata keys were renamed from `document_index` to `document_id`, but `create_or_load_chunk_embeddings` returns cached `chunk_metadata` from Redis/S3 without checking its format. | Any user whose index was built before commit `9ad67bc` gets `KeyError: 'document_id'` on their first semantic search. |
| R2 | `rag/storage/storage.py:21-26`, `rag/custom_types/custom_types.py:17-22` | The S3 client is built from per-user `aws_access_key_id` / `aws_secret_access_key` stored on the `User` model. | Secret keys live in application data and memory. `User` is also registered with the TypeConverter (`storage.py:32`), so it can be serialized with its secrets. This conflicts with the roadmap's "IAM role, no hardcoded credentials". |
| R3 | `rag/search/hybrid_search.py:17-21`, `rag/inverted_index/inverted_index.py:78-88`, `rag/semantic_index/semantic_index.py:104-105` | `HybridSearch.__init__` loads indexes, but if storage is empty both `InvertedIndex.load()` and `create_or_load_chunk_embeddings()` build and save the index. | Violates "queries must never trigger indexing". A query can trigger a full build and write to S3. |

### 1.2 Medium

| # | Location | Issue | Impact |
|---|----------|-------|--------|
| R4 | `rag/storage/storage.py:43,51,59,68` | Keys are built as `f"{user_id}/{name}"`. | Violates the required `users/{user_id}/` layout. Changing it later requires migrating stored data. |
| R5 | `rag/storage/storage.py:16` | `redis.Redis(...)` has no `socket_timeout` / `socket_connect_timeout`. | An unreachable (not refusing) Redis can hang calls instead of failing over to S3, which breaks "Redis failures must never be fatal". |
| R6 | `rag/storage/storage.py:64-73` | On a cache miss, data loaded from S3 is not written back to Redis. | Redis is only filled on upload, so after TTL expiry every load hits S3. |
| R7 | `rag/semantic_index/semantic_index.py:70-83` | `build_chunk_embeddings` returns `None` on upload failure, after in-memory state has already been updated. | Callers can't tell a failed save from success. Memory and S3 can silently diverge. |
| R8 | `rag/inverted_index/inverted_index.py:64-75` | `save()` logs and swallows every storage error without returning or raising anything. | A failed save is invisible to the caller. |
| R9 | `rag/semantic_index/semantic_index.py:14,44` | `docmap` is never cleared between builds. | Documents that are deleted can still be returned by search for the lifetime of the `SemanticIndex` instance. |
| R10 | `rag/custom_types/db_types.py` vs `rag/custom_types/custom_types.py` | `DbUser` (mirrors the DB) and `User` (used by `Storage`) have unrelated fields. `User` has AWS/bucket fields the database doesn't. | Two diverging user models, with no defined mapping from the DB row to what `Storage` needs. |
| R11 | `rag/rag/rag.py:45` | `generate` is typed `Callable[[str, str], Awaitable[str]]`, but `llm_ollama` and `llm_bedrock` return `str \| None`. | The type hints don't match the providers. `None` is handled at runtime but hidden from the type checker. |
| R12 | `rag/llm/bedrock.py:16` | A new `boto3` Bedrock client is created on every call. | Unnecessary per-request overhead (credential resolution, connection setup). |

### 1.3 Low

| # | Location | Issue |
|---|----------|-------|
| R13 | `rag/storage/storage.py:1`, `rag/semantic_index/semantic_index.py:1`, `rag/llm/bedrock.py:5` | `from botocore.client import logging` only works because botocore imports `logging` itself. It should be the standard library `logging` module with a module logger. |
| R14 | `rag/storage/storage.py:72` | Uses `print` instead of the logger for S3 load failures. |
| R15 | `rag/llm/ollama.py:19` | Catches bare `Exception`. |
| R16 | `rag/semantic_index/semantic_index.py:151` | Linear `filter` over all chunk metadata for each result (O(results × chunks)). |
| R17 | `rag/semantic_index/semantic_index.py:165-196`, `rag/rag/rag.py` (end of file) | Long runs of trailing blank lines. `ruff format --check` will fail. |
| R18 | `rag/server.py` | Empty stub (`if __name__ == "__main__": pass`). |
| R19 | `rag/` | No `pyproject.toml` / `uv.lock`. The `uv` + Ruff workflow in `CLAUDE.md` can't run (`uv run ruff` fails). Dependencies are in the root `requirements.txt`, which lacks `moto`, `pytest-asyncio`, `ollama`. |
| R20 | `rag/test/` | No unit tests for `InvertedIndex`, `SemanticIndex`, `HybridSearch`, chunking helpers, `llm_ollama`, or `llm_bedrock`. They are only exercised through one e2e test. |

---

## 2. API Service (Go)

### 2.1 High

| # | Location | Issue | Impact |
|---|----------|-------|--------|
| A1 | `api/cmd/api/logger.go`, `api/cmd/api/main.go` | `initLogger` is never called. | The service has no structured logging. |
| A2 | `api/cmd/api/logger.go:12,18,44` | The log `*os.File` is opened but never returned, so it can't be closed. Output goes through a `bufio.Writer` that must be flushed, and `main` exits via `os.Exit`, which skips defers. | Once wired in: file handle leak, and up to one buffer (8 KB by default) of logs lost on every exit or crash. |
| A3 | `api/cmd/api/server.go:24-29` | `http.Server` has no `ReadHeaderTimeout`, `ReadTimeout`, `WriteTimeout`, or `IdleTimeout`. | Open to slowloris and connection exhaustion. golangci-lint (gosec G112) will flag it. |

### 2.2 Medium

| # | Location | Issue | Impact |
|---|----------|-------|--------|
| A4 | `api/internal/config/config.go` | The file is not gofmt'd (mixed tabs and spaces). | `gofmt -l .` fails, which blocks the pre-commit check in `CLAUDE.md`. |
| A5 | `api/internal/config/config.go:28-42` | `PORT` and `CUSTOM_BUFFER_SIZE` are parsed but not range-checked. | Out-of-range ports (e.g. `99999`) or non-positive sizes are accepted and fail later, far from the cause. |
| A6 | `api/cmd/api/main.go:47-50` | The signal context's stop function isn't called when the first signal arrives. | A second Ctrl+C / SIGTERM can't force-quit during a slow shutdown. |
| A7 | `api/cmd/api/logger.go:10` | Logs go to a local file instead of stdout. | In containers (EKS/CloudWatch), file logs aren't collected by default. |
| A8 | `api/` | No tests at all (`config.Load`, `run`, `server`). | Violates "every meaningful change should include appropriate tests". |

### 2.3 Low

| # | Location | Issue |
|---|----------|-------|
| A9 | `api/cmd/api/server.go:16,33`, `api/cmd/api/main.go:26,33` | `server.cancel` is stored but never used. `run` accepts `cancel` only to pass it along. |
| A10 | `api/internal/config/config.go:39` | Error message says `"invalid CUSTOM BUFFER SIZE"` instead of the actual variable name `CUSTOM_BUFFER_SIZE`. |
| A11 | `api/cmd/api/replace_attributes.go` | Empty file (only `package main`). |
| A12 | `api/internal/domain/domain.go:24` | Empty "types between api and pubsub" section. |
| A13 | `api/cmd/api/main.go`, `api/cmd/api/server.go` | Comment typos: "depedning", "errirs", "mathc", "intul". |

---

## 3. Database & Local Stack

| # | Severity | Location | Issue | Impact |
|---|----------|----------|-------|--------|
| D1 | Medium | `migrations/alembic/versions/811ecb922478_*.py:40` | The `documents.user_id` FK has no `ON DELETE CASCADE`. Cascade exists only at the ORM level (`models/user.py`). | Deleting a user with raw SQL, or from any non-SQLAlchemy client such as the Go API, fails with an FK violation or requires manual cleanup. |
| D2 | Medium | `docker-compose.yml` | Uses `postgres:15-alpine`, which doesn't ship the pgvector extension. | The planned pgvector work can't run on the local stack as configured. |
| D3 | Low | `models/base.py:1` | Imports `declarative_base` from `sqlalchemy.ext.declarative`, which is deprecated in SQLAlchemy 2.0. | Deprecation warning; will break in a future SQLAlchemy release. |
| D4 | Low | `docker-compose.yml` | Hardcoded `POSTGRES_PASSWORD: password`. | Acceptable locally, but it must not carry over to any shared environment. |

---

## 4. Summary

| Area | High | Medium | Low |
|------|------|--------|-----|
| RAG (Python) | 3 | 9 | 8 |
| API (Go) | 3 | 5 | 5 |
| Database & Stack | 0 | 2 | 2 |
| **Total** | **6** | **16** | **15** |

Suggested order to address: R1, R3, A1/A2, A3, R2, R4, A4.
