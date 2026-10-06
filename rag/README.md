# BlightSanest RAG Service

The Python retrieval and generation service of BlightSanest. It builds per-user search indexes over a user's documents, runs hybrid search (BM25 + semantic) over them, and asks an LLM to answer a query using only the retrieved documents.

It is domain-agnostic: every document arrives as a plain string (`Document(id, content)`), whatever its domain.

> **Status**: core pipeline implemented and tested end-to-end with a mocked LLM. Not yet exposed as a service: `server.py` is an empty stub and the gRPC contract with the Go API is still planned. See `../BlightSanest_Progress_Roadmap.md` and `../CODE_ISSUES.md`.

---

## Where It Fits

```
API (Go) ──gRPC (planned)──▶ RAG (Python) ──▶ S3 (source of truth) / Redis (cache)
```

- The API is the only caller. RAG never talks to PubSub.
- In v1, RAG has **no database access**. In v2 it may read the shared index, but never writes. All database writes go through the API.
- Every user has **isolated indexes**; there is no global or cross-user index.

---

## Pipeline

```
ingestion ─▶ HybridSearch.load_or_empty(storage, model)
          ─▶ build(documents) / remove_documents(ids)   (BM25 + embeddings, in memory)
          ─▶ save() ─▶ new snapshot + manifest ─▶ Storage (S3 + Redis)

query ─▶ HybridSearch.load(storage, model) ─▶ BM25 + semantic ─▶ RRF ─▶ top documents
      ─▶ RAG.rag(query, documents) ─▶ LLM ─▶ RagResponse{status, response}
```

1. **Indexing** (ingestion only) adds, replaces, or removes documents in both indexes, then `save()` writes them as one snapshot. Only changed documents are re-embedded.
2. **Search** loads the user's saved snapshot and never builds one: with no saved index, `HybridSearch.load()` raises `IndexNotBuiltError`. It scores documents with BM25 and with embedding similarity, then fuses both rankings with Reciprocal Rank Fusion.
3. **Generation** sends the query and the retrieved documents to the injected LLM, which must answer in JSON with `status` (`found` / `not found`) and `response`.

---

## Modules

| Path | What it does |
|------|--------------|
| `inverted_index/` | `InvertedIndex(docmap)`: BM25 from scratch (k1 = 1.5, b = 0.75). Token → doc IDs, term frequencies, doc lengths. `build()` (re-adding a `doc_id` replaces it), `remove_document()`, `bm25_search()`, and `export_parts()` / `restore_parts()` for the snapshot. No storage access of its own. |
| `semantic_index/` | `SemanticIndex(docmap, embedding_model=...)`: sentence-transformer embeddings over chunks of 4 sentences with 1 overlapping (sentences split on `.`, `!`, `?` and line breaks). The model is always injected: `create_embedding_model(config)` loads the one named by `Config.embedding_model_name` once at startup. Chunk metadata: `document_id`, `chunk_index`, `total_chunks`. A document's score is its best chunk's cosine similarity, and its result carries that chunk's metadata; chunks resolve to documents through `docmap` by `document_id`. `build_chunk_embeddings()` is incremental: it drops the given documents' chunks and embeds only those documents. `remove_document()`, and `export_parts()` / `restore_parts()` (which rejects embeddings and metadata of different lengths). A blank query or an empty index returns `[]`. |
| `search/` | `HybridSearch`: owns one user's two indexes, their shared docmap, and their persistence (see Storage Layout). `load(storage, embedding_model)` for queries, `load_or_empty(storage, embedding_model)` + `build()` / `remove_documents()` + `save()` for ingestion. Raises `IndexNotBuiltError` (no saved index) or `CorruptIndexError` (snapshot parts missing or inconsistent). Fuses BM25 and semantic ranks with RRF (k = 60) over the union of results, and returns at most `limit` results (default 50). A blank query returns `[]`. |
| `rag/` | `RAG`: prompt construction and JSON response parsing. The LLM function is injected via the constructor; `RAG` never selects a provider. Replies may be wrapped in ```` ```json ```` fences; `status` must be `found` or `not found`. Every invalid reply raises `ValueError`. |
| `llm/` | LLM providers with the signature `async (user_content, system_content, *, settings) -> str \| None`. `ollama_provider.py` → `llm_ollama(..., host=, model=)` (development); `bedrock.py` → `llm_bedrock(..., region=, model_id=)` (production, Converse API). Bind the settings from `Config` with `functools.partial` at startup so `RAG` gets a plain `(user, system)` callable. Errors are logged and returned as `None`. |
| `storage/` | `Storage(user, bucket_name, s3_client, redis_connection)`: `upload_data(name, data)` writes to S3 then Redis (TTL 3600s); `load_data(name)` reads Redis first, falls back to S3; `delete_data(name)`. `cache=False` bypasses Redis. `load_data` returns `None` only when the object doesn't exist; any other S3 error raises. Redis errors and unreadable cached values are logged and never fatal, and a failed Redis write drops the cached key. `clients.py` creates the shared S3 and Redis clients once from `Config`. |
| `type_converter/` | `TypeConverter`: MessagePack serialization with a type registry for set, tuple, Counter, OrderedDict, defaultdict, numpy arrays, and registered Pydantic models. Serializing an unregistered model, or deserializing an unknown type tag, raises `TypeError`. |
| `config/` | `Config` (bucket, region, Redis host/port) and `load_config()`, which reads it once from environment variables. |
| `custom_types/` | Pydantic models: `Document`, `User` (just `id`), `RagResponse` (`custom_types.py`). |
| `helpers/` | Tokenizing (NLTK, lowercased, English stopwords and punctuation-only tokens dropped), cosine similarity, chunking, RRF score, JSON parsing. |
| `constants/` | `BM25_K1`, `BM25_B`, `SEARCH_LIMIT`. |
| `server.py` | Placeholder for the gRPC server (empty). |
| `test/` | pytest suite (see below). |

---

## Storage Layout

Each user's index is saved as a versioned snapshot of MessagePack blobs, plus a manifest naming the live one:

```
{bucket}/users/{user_id}/manifest                      {"version": ..., "previous": ...}
{bucket}/users/{user_id}/snapshots/{version}/docmap
{bucket}/users/{user_id}/snapshots/{version}/inverted_index
{bucket}/users/{user_id}/snapshots/{version}/term_frequencies
{bucket}/users/{user_id}/snapshots/{version}/doc_lengths
{bucket}/users/{user_id}/snapshots/{version}/chunk_embeddings
{bucket}/users/{user_id}/snapshots/{version}/chunk_metadata
```

`save()` writes every part under a new random `{version}`, then switches the manifest to it, so a reader sees either the old snapshot or the new one, never a mix. A failed save raises and leaves the previous snapshot live (and deletes the parts it had written). The live snapshot and the one before it are kept; older ones are deleted. Saves for one user must not run concurrently: the last manifest write wins.

S3 is the source of truth; Redis is a disposable hot cache and uses the same keys. `Storage._key()` builds every key, so S3 and Redis always agree. Snapshot parts never change once written, so caching them is always safe; the manifest is the one mutable key and is never cached.

S3 reports a missing key as `NoSuchKey` only if the reader has `s3:ListBucket` on the bucket; without it, S3 answers `AccessDenied`, which `Storage` treats as a failure. The service's IAM role therefore needs `s3:ListBucket` as well as `GetObject`, `PutObject`, and `DeleteObject`.

No AWS keys appear in code or on `User`. The S3 client uses boto3's default credential chain: environment variables or `~/.aws` locally, the pod's IAM role on EKS. Per-user isolation comes from the key prefix. If your `~/.aws` profile was set up with `aws login`, boto3 needs the `botocore[crt]` extra to read it.

---

## Configuration

| Variable | `Config` field → used by | Default |
|----------|---------|---------|
| `S3_BUCKET` | `bucket_name` → `Storage` | none (required) |
| `AWS_REGION` | `region` → S3 client and `llm_bedrock` | `us-east-1` |
| `REDIS_HOST` | `redis_host` → Redis client | `localhost` |
| `REDIS_PORT` | `redis_port` → Redis client | `6379` |
| `SENTENCE_TRANSFORMERS_MODEL_NAME` | `embedding_model_name` → `create_embedding_model()` | `all-MiniLM-L6-v2` |
| `OLLAMA_HOST` | `ollama_host` → `llm_ollama` | `http://localhost:11434` |
| `OLLAMA_MODEL` | `ollama_model` → `llm_ollama` | `gemma3` |
| `BEDROCK_MODEL_ID` | `bedrock_model_id` → `llm_bedrock` | `anthropic.claude-opus-5-5` |

`load_config()` in `config/` is the only code in `rag/` that reads environment variables; everything else gets its settings from the `Config` it returns.

The Redis client uses 2-second connect and read timeouts and no retries, so a down Redis costs at most 2 seconds before the S3 fallback.

---

## Usage

```python
import asyncio

from functools import partial

from config.config import load_config
from custom_types.custom_types import Document, User
from llm.ollama_provider import llm_ollama
from rag.rag import RAG
from search.hybrid_search import HybridSearch
from semantic_index.semantic_index import create_embedding_model
from storage.clients import create_redis_client, create_s3_client
from storage.storage import Storage

# once at startup: config, shared clients, the embedding model, the LLM
config = load_config()  # needs S3_BUCKET
s3_client = create_s3_client(config)
redis_connection = create_redis_client(config)
embedding_model = create_embedding_model(config)
generate = partial(llm_ollama, host=config.ollama_host, model=config.ollama_model)

# per user
user = User(id="user_123")
storage = Storage(user, config.bucket_name, s3_client, redis_connection)
docs = [Document(id="doc1", content="Today I felt anxious about my presentation.")]

# ingestion: add or replace documents, then save one snapshot
index = HybridSearch.load_or_empty(storage, embedding_model)
index.build(docs)
index.save()

# query: load the saved snapshot (raises IndexNotBuiltError if there is none)
search = HybridSearch.load(storage, embedding_model)
results = search.rrf_search("felt anxious")

retrieved = [Document(id=r["doc_id"], content=r["content"]) for r in results]
answer = asyncio.run(RAG(generate).rag("How did I feel?", retrieved))
print(answer.status, answer.response)
```

---

## Development

Requirements: Python 3.12 (repo-root `.python-version`) and [uv](https://docs.astral.sh/uv/). This folder is a member of the repo's uv workspace. Its dependencies are declared in `rag/pyproject.toml`, but they are locked in the root `uv.lock` and installed into the shared root `.venv`. Test and lint tools (pytest, pytest-asyncio, moto, ruff) are in the `dev` group.

```bash
uv sync                  # from the repo root: installs all members into .venv
cd rag
uv add <package>         # add a runtime dependency to rag
uv add --dev <package>   # add a test/lint dependency to rag
```

Don't run `uv sync` inside `rag/`: it would uninstall the `migrations` packages from the shared environment. `uv run` is safe here.

Importing `helpers` downloads the NLTK `punkt_tab` and `stopwords` data on first run, so it needs network access. The tests load the default embedding model (`all-MiniLM-L6-v2`) once per run, through the session-scoped `embedding_model` fixture in `test/conftest.py`.

Local services:

```bash
docker-compose up -d redis ollama   # from the repo root
```

### Tests

```bash
cd rag
uv run pytest
uv run pytest test/test_rag.py::TestRagEnd2End::test_full_pipeline -q
```

140 tests in total:

| File | Tests | Covers |
|------|-------|--------|
| `test/test_type_converter.py` | 25 | Round trips for each supported type, nested structures, `defaultdict(Counter)` and pydantic models (the stored index shapes); unregistered models and unknown type tags raise |
| `test/test_helpers.py` | 21 | `cosine_similarity` returns plain floats, `base_chunk` windows and its argument check, `semantic_chunk` (including line breaks), `tokenize` (stopwords, punctuation), fence stripping in `parse_json` |
| `test/test_inverted_index.py` | 20 | `build()`, IDF / length-normalized TF / BM25 against hand-computed values, ranking and `limit`, export → restore round trip, replacing and removing documents |
| `test/test_storage.py` | 18 | Model registration, `users/{user_id}/` keys, upload and load paths, Redis-failure tolerance, only a missing key reads as `None`, corrupt cache falls back to S3, `cache=False`, failed Redis writes drop the key, `delete_data` |
| `test/test_search.py` | 16 | RRF `limit`, blank queries, content from the shared docmap; snapshots on a real `Storage` over moto S3: the query path never builds, save → load, update and delete, pruning, failed saves keep the old snapshot, corrupt snapshots, S3 outages raise |
| `test/test_semantic_index.py` | 13 | Incremental `build_chunk_embeddings` (only changed documents re-embedded), `remove_document`, export/restore and misaligned parts, best-chunk metadata, blank query and empty index (stub model) |
| `test/test_llm.py` | 9 | Mocked `llm_ollama` and `llm_bedrock`: request shape and the host / region they are given, and `None` on model, connection, throttling and malformed-response errors |
| `test/test_rag_parsing.py` | 9 | Valid and fenced LLM replies accepted; invalid replies raise `ValueError` |
| `test/test_config.py` | 8 | `load_config()` defaults, environment, invalid values; client factories use the config; the Redis client fails fast (no retries) |
| `test/test_rag.py` | 1 | End-to-end: the query path refuses an unbuilt index → ingestion build + save → fresh load → RRF search → `RAG` with a mocked LLM |

S3 is mocked with moto (`mock_aws`). Async tests use pytest-asyncio in strict mode (`pytest.ini`). Storage tests use mock clients. The e2e test uses a real Redis at `localhost:6379` if one is running, and otherwise falls back to (mocked) S3. It runs as a new random user each time, so cached keys from earlier runs can't stand in for the build, and it deletes its own Redis keys afterwards.
