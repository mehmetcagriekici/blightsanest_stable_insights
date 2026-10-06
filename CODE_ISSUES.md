# BlightSanest: Known Code Issues

**Last Updated**: October 6, 2026
**Scope**: `api/` (Go), `rag/` (Python), `models/` + `migrations/`

The main tables track serious problems:
- code that breaks,
- security risks,
- data loss,
- violations of the hard constraints in `CLAUDE.md`.

Lower-severity logic inconsistencies and performance problems are listed separately in section 1.1. Pure style and typing issues are out of scope. IDs are stable and are not renumbered when items are removed.

---

## 1. RAG Service (Python)

No open serious issues.

### 1.1 RAG: lower severity

| # | Location | Issue | Impact |
|---|----------|-------|--------|
| R35 | `rag/helpers/helpers.py:96-106` | Chunks are 4 sentences, but all-MiniLM-L6-v2 only reads the first 256 word pieces. Line breaks now also split sentences, so unpunctuated line-based entries are chunked. | Partial: a single long sentence can still be truncated. Token-aware chunking is deferred because it changes every stored index. |
| R43 | `rag/type_converter/type_converter.py:74-83` | Embeddings are stored as `tolist()` Python floats in msgpack. | Several times larger and slower than storing `arr.tobytes()`. Deferred: no measured need yet. |
| R45 | `rag/custom_types/db_types.py`, `rag/storage/storage.py:26,32,46`, `rag/search/hybrid_search.py:65` | Leftovers from the plan for RAG to read the database, now deferred to v2. `DbUser` and `DbDocument` are never used, and `DbUser` copies the `users` row, including `hashed_password`, into the RAG codebase. `Storage.database_user` holds a plain `User` (just `id`) but is named as if it were a database row. | No runtime effect. The models suggest RAG reads the database, which v1 forbids, and put a password-hash field in a service that should never see one. The field name misleads readers. |

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
| RAG (Python) | 0 |
| RAG (Python), lower severity | 4 |
| API (Go) | 2 |
| Database | 1 |
| **Total** | **7** |

Suggested order to address: A3, A2, D1, then section 1.1.
