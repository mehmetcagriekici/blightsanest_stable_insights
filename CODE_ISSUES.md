# BlightSanest: Known Code Issues

**Last Updated**: September 2026
**Scope**: `api/` (Go), `rag/` (Python), `models/` + `migrations/`

This file tracks only serious problems:
- code that breaks,
- security risks,
- data loss,
- violations of the hard constraints in `CLAUDE.md`.

Style, typing, performance, and missing tests are out of scope. IDs are stable and are not renumbered when items are removed.

---

## 1. RAG Service (Python)

| # | Location | Issue | Impact |
|---|----------|-------|--------|
| R2 | `rag/storage/storage.py:21-26`, `rag/custom_types/custom_types.py:17-22` | The S3 client is built from per-user `aws_access_key_id` / `aws_secret_access_key` stored on the `User` model. | Secret keys live in application data and memory. `User` is registered with the TypeConverter (`storage.py:32`), so it can be serialized with its secrets. Conflicts with the roadmap's "IAM role, no hardcoded credentials". |
| R3 | `rag/search/hybrid_search.py:17-21`, `rag/inverted_index/inverted_index.py:84-98`, `rag/semantic_index/semantic_index.py:110-111` | `HybridSearch.__init__` loads indexes, but if storage is empty both `InvertedIndex.load()` and `create_or_load_chunk_embeddings()` build and save the index. | Violates the hard constraint "queries must never trigger indexing". A query can trigger a full build and write to S3. |
| R4 | `rag/storage/storage.py:43,51,59,68` | Keys are built as `f"{user_id}/{name}"`. | Violates the required `users/{user_id}/` layout. Changing it after real data exists requires migrating stored objects. |
| R5 | `rag/storage/storage.py:16` | `redis.Redis(...)` has no `socket_timeout` / `socket_connect_timeout`. | An unreachable (not refusing) Redis hangs calls instead of failing over to S3. This breaks "Redis failures must never be fatal". |
| R7 | `rag/semantic_index/semantic_index.py:76-91` | `build_chunk_embeddings` logs and returns `None` when the upload fails, after in-memory state has already been updated. | A failed write to S3, the source of truth, goes unnoticed. The index then exists only in memory and is lost when the process ends. |
| R8 | `rag/inverted_index/inverted_index.py:63-81` | `InvertedIndex.save()` logs and swallows every storage error without returning or raising anything. | Same as R7, for the BM25 index. |

---

## 2. API Service (Go)

| # | Location | Issue | Impact |
|---|----------|-------|--------|
| A2 | `api/cmd/api/logger.go:12,18,44` | The log `*os.File` is opened but never returned, so it can't be closed. Output goes through a `bufio.Writer` that must be flushed, and `main` exits via `os.Exit`, which skips defers. | Once the logger is wired in: a leaked file handle, and up to one buffer (8 KB by default) of logs lost on every exit or crash. That includes the logs explaining the crash. |
| A3 | `api/cmd/api/server.go:24-29` | `http.Server` has no `ReadHeaderTimeout`, `ReadTimeout`, `WriteTimeout`, or `IdleTimeout`. | Open to slowloris and connection exhaustion. |

---

## 3. Database

| # | Location | Issue | Impact |
|---|----------|-------|--------|
| D1 | `migrations/alembic/versions/811ecb922478_*.py:40` | The `documents.user_id` FK has no `ON DELETE CASCADE`. Cascade exists only at the SQLAlchemy ORM level (`models/user.py`). | All writes go through the Go API, which doesn't use the ORM, so deleting a user who has documents fails with an FK violation. |

---

## 4. Summary

| Area | Open |
|------|------|
| RAG (Python) | 6 |
| API (Go) | 2 |
| Database | 1 |
| **Total** | **9** |

Suggested order to address: R3, R2, R7/R8, R5, A3, A2, R4, D1.
