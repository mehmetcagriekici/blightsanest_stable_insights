# BlightSanest: Known Code Issues

**Last Updated**: September 30, 2026
**Scope**: `api/` (Go), `rag/` (Python), `models/` + `migrations/`

The main tables track serious problems:
- code that breaks,
- security risks,
- data loss,
- violations of the hard constraints in `CLAUDE.md`.

Lower-severity logic inconsistencies and performance problems are listed separately in section 1.1. Pure style and typing issues are out of scope. IDs are stable and are not renumbered when items are removed.

---

## 1. RAG Service (Python)

| # | Location | Issue | Impact |
|---|----------|-------|--------|
| R3 | `rag/search/hybrid_search.py:11-23`, `rag/inverted_index/inverted_index.py:84-98`, `rag/semantic_index/semantic_index.py:103-120` | `HybridSearch.__init__` loads indexes, but if storage is empty both `InvertedIndex.load()` and `create_or_load_chunk_embeddings()` build and save the index. | Violates the hard constraint "queries must never trigger indexing". A query can trigger a full build and write to S3. |
| R7 | `rag/semantic_index/semantic_index.py:77-100` | `build_chunk_embeddings` logs and returns `None` when the upload fails, after in-memory state has already been updated. | A failed write to S3, the source of truth, goes unnoticed. The index then exists only in memory and is lost when the process ends. |
| R8 | `rag/inverted_index/inverted_index.py:63-81` | `InvertedIndex.save()` logs and swallows every storage error without returning or raising anything. | Same as R7, for the BM25 index. |
| R27 | `rag/storage/storage.py:64-71` | If the S3 put succeeds but the Redis `set` fails, the error is logged and the old Redis value is left in place. Reads check Redis first. | For up to `redis_ttl` (1 hour), reads return data that no longer matches S3, the source of truth. The failed-set path should delete the key. |
| R29 | `rag/storage/storage.py:74-94` | `load_data` returns `None` for every `ClientError` (AccessDenied, throttling, etc.), not just NoSuchKey. `BotoCoreError` (e.g. connection failure) isn't caught. A corrupt Redis value makes msgpack raise instead of falling back to S3. | Combined with R3, a transient S3 error during a query looks like "index not built": the index is rebuilt from whatever `documents` were passed and overwrites S3. If that list was partial, data is lost. |
| R30 | `rag/semantic_index/semantic_index.py:77-81`, `rag/inverted_index/inverted_index.py:63-68` | Index parts are uploaded as separate keys with no atomicity. If `chunk_embeddings` uploads and `chunk_metadata` fails, or only some of the four inverted-index keys upload, storage holds a mix of old and new parts. | `search_chunks` pairs embeddings with metadata by position (`self.chunk_metadata[i]`), so it raises `IndexError` or credits scores to the wrong document. Makes R7/R8 worse than "lost on restart". |
| R32 | `rag/semantic_index/semantic_index.py:103-107`, `rag/inverted_index/inverted_index.py:66,84-98`, `rag/search/hybrid_search.py:80-84` | The two indexes use different docmaps. The inverted index persists its `docmap` to S3; the semantic index builds its own from the `documents` passed on every query and never persists it. `rrf_search` takes content from the semantic side if present, otherwise from the stored docmap. | The same `doc_id` can come back with different content depending on which search found it. Every query must also load every document's full content just to construct the indexes. |

### 1.1 RAG: lower severity

| # | Location | Issue | Impact |
|---|----------|-------|--------|
| R35 | `rag/helpers/helpers.py:93-104` | Chunks are 4 sentences, but all-MiniLM-L6-v2 only reads the first 256 word pieces. Line breaks now also split sentences, so unpunctuated line-based entries are chunked. | Partial: a single long sentence can still be truncated. Token-aware chunking is deferred because it changes every stored index. |
| R44 | `rag/semantic_index/semantic_index.py:14` | The model is loaded at import time from `SENTENCE_TRANSFORMERS_MODEL_NAME`, with no check. `SentenceTransformer(None)` doesn't fail cleanly. | If the variable is unset, importing `semantic_index` (and anything that imports `search`) crashes with `AttributeError: 'NoneType' object has no attribute 'parameters'`, which doesn't name the missing variable. The env read also bypasses `rag/config/`. |
| R43 | `rag/type_converter/type_converter.py:74-83` | Embeddings are stored as `tolist()` Python floats in msgpack. | Several times larger and slower than storing `arr.tobytes()`. Deferred: no measured need yet. |

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
| RAG (Python) | 7 |
| RAG (Python), lower severity | 3 |
| API (Go) | 2 |
| Database | 1 |
| **Total** | **13** |

Suggested order to address: R3, R29, R7/R8/R30, R27, R32, A3, A2, D1, then section 1.1.
