# BlightSanest RAG Service

The Python retrieval and generation service of BlightSanest. It builds per-user search indexes over a user's documents, runs hybrid search (BM25 + semantic) over them, and asks an LLM to answer a query using only the retrieved documents.

It is domain-agnostic: every document arrives as a plain string (`Document(id, content)`), whatever its domain.

> **Status**: core pipeline implemented and tested end-to-end with a mocked LLM. Not yet exposed as a service: `server.py` is an empty stub and the gRPC contract with the Go API is still planned. See `../BlightSanest_Progress_Roadmap.md` and `../CODE_ISSUES.md`.

---

## Where It Fits

```
API (Go) ──gRPC (planned)──▶ RAG (Python) ──▶ S3 (source of truth) / Redis (cache)
                                  │
                                  └─ read-only ─▶ Aurora + pgvector (planned)
```

- The API is the only caller. RAG never talks to PubSub.
- RAG is **read-only** toward the database; all writes go through the API.
- Every user has **isolated indexes**; there is no global or cross-user index.

---

## Pipeline

```
documents ─▶ InvertedIndex.build()/save() ─────────┐
          └▶ SemanticIndex.build_chunk_embeddings() ┴─▶ Storage (S3 + Redis)

query ─▶ HybridSearch (load indexes) ─▶ BM25 + semantic ─▶ RRF ─▶ top documents
      ─▶ RAG.rag(query, documents) ─▶ LLM ─▶ RagResponse{status, response}
```

1. **Indexing** builds the BM25 index and chunk embeddings, then saves them through `Storage`.
2. **Search** loads the user's indexes, scores documents with BM25 and with embedding similarity, then fuses both rankings with Reciprocal Rank Fusion.
3. **Generation** sends the query and the retrieved documents to the injected LLM, which must answer in JSON with `status` (`found` / `not found`) and `response`.

> Indexes are meant to be built only at ingestion time. Currently `HybridSearch.__init__` still builds and saves an index when none exists in storage (see `CODE_ISSUES.md`, R3).

---

## Modules

| Path | What it does |
|------|--------------|
| `inverted_index/` | `InvertedIndex`: BM25 from scratch (k1 = 1.5, b = 0.75). Token → doc IDs, term frequencies, doc lengths, docmap. `build()`, `save()`, `load()`, `bm25_search()`. |
| `semantic_index/` | `SemanticIndex(storage)`: sentence-transformer embeddings over chunks of 4 sentences with 1 overlapping (sentences split on `.`, `!`, `?` and line breaks). The model named by `SENTENCE_TRANSFORMERS_MODEL_NAME` is loaded once per process, when the module is imported. Chunk metadata: `document_id`, `chunk_index`, `total_chunks`. A document's score is its best chunk's cosine similarity, and its result carries that chunk's metadata; chunks resolve to documents through `docmap` by `document_id`. A blank query returns `[]`. |
| `search/` | `HybridSearch(storage, documents)`: loads both indexes for a user, fuses BM25 and semantic ranks with RRF (k = 60) over the union of results, and returns at most `limit` results (default 50). A blank query returns `[]`. |
| `rag/` | `RAG`: prompt construction and JSON response parsing. The LLM function is injected via the constructor; `RAG` never selects a provider. Replies may be wrapped in ```` ```json ```` fences; `status` must be `found` or `not found`. Every invalid reply raises `ValueError`. |
| `llm/` | LLM providers with the signature `async (user_content, system_content) -> str \| None`. `ollama_provider.py` → `llm_ollama` (development); `bedrock.py` → `llm_bedrock` (production, Converse API). Errors are logged and returned as `None`. |
| `storage/` | `Storage(user, bucket_name, s3_client, redis_connection)`: `upload_data(name, data)` writes to S3 then Redis (TTL 3600s); `load_data(name)` reads Redis first, falls back to S3. Redis errors are logged and never fatal. `clients.py` creates the shared S3 and Redis clients once from `Config`. |
| `type_converter/` | `TypeConverter`: MessagePack serialization with a type registry for set, tuple, Counter, OrderedDict, defaultdict, numpy arrays, and registered Pydantic models. Serializing an unregistered model, or deserializing an unknown type tag, raises `TypeError`. |
| `config/` | `Config` (bucket, region, Redis host/port) and `load_config()`, which reads it once from environment variables. |
| `custom_types/` | Pydantic models: `Document`, `User` (just `id`), `RagResponse` (`custom_types.py`); `DbUser`, `DbDocument` mirroring DB rows (`db_types.py`). |
| `helpers/` | Tokenizing (NLTK, English stopwords), cosine similarity, chunking, RRF score, JSON parsing. |
| `constants/` | `BM25_K1`, `BM25_B`, `SEARCH_LIMIT`. |
| `server.py` | Placeholder for the gRPC server (empty). |
| `test/` | pytest suite (see below). |

---

## Storage Layout

Each object is a MessagePack blob keyed per user, in both S3 and Redis:

```
{bucket}/users/{user_id}/inverted_index
{bucket}/users/{user_id}/docmap
{bucket}/users/{user_id}/term_frequencies
{bucket}/users/{user_id}/doc_lengths
{bucket}/users/{user_id}/chunk_embeddings
{bucket}/users/{user_id}/chunk_metadata
```

S3 is the source of truth; Redis is a disposable hot cache and uses the same `users/{user_id}/{name}` keys. `Storage._key()` builds every key, so S3 and Redis always agree.

No AWS keys appear in code or on `User`. The S3 client uses boto3's default credential chain: environment variables or `~/.aws` locally, the pod's IAM role on EKS. Per-user isolation comes from the key prefix. If your `~/.aws` profile was set up with `aws login`, boto3 needs the `botocore[crt]` extra to read it.

---

## Configuration

| Variable | Used by | Default |
|----------|---------|---------|
| `S3_BUCKET` | `load_config()` → S3 bucket | none (required) |
| `AWS_REGION` | `load_config()` → S3 client region | `us-east-1` |
| `REDIS_HOST` | `load_config()` → Redis client | `localhost` |
| `REDIS_PORT` | `load_config()` → Redis client | `6379` |
| `OLLAMA_HOST` | `llm_ollama` | `http://localhost:11434` |
| `AWS_REGION_NAME` | `llm_bedrock` | none (boto3's own region lookup) |
| `BEDROCK_MODEL_ID` | `llm_bedrock` | none (must be set) |
| `SENTENCE_TRANSFORMERS_MODEL_NAME` | `semantic_index` (model loaded once at import) | none (required, e.g. `all-MiniLM-L6-v2`) |

The Redis client uses 2-second connect and read timeouts. The Ollama model defaults to `gemma3`.

---

## Usage

```python
import asyncio

from config.config import load_config
from custom_types.custom_types import Document, User
from llm.ollama_provider import llm_ollama
from rag.rag import RAG
from search.hybrid_search import HybridSearch
from storage.clients import create_redis_client, create_s3_client
from storage.storage import Storage

# once at startup: config and shared clients
config = load_config()  # needs S3_BUCKET
s3_client = create_s3_client(config)
redis_connection = create_redis_client(config)

# per user
user = User(id="user_123")
storage = Storage(user, config.bucket_name, s3_client, redis_connection)
docs = [Document(id="doc1", content="Today I felt anxious about my presentation.")]

search = HybridSearch(storage, docs)
results = search.rrf_search("felt anxious")

retrieved = [Document(id=r["doc_id"], content=r["content"]) for r in results]
answer = asyncio.run(RAG(llm_ollama).rag("How did I feel?", retrieved))
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

Importing `helpers` downloads the NLTK `punkt_tab` and `stopwords` data on first run, so it needs network access. Importing `semantic_index` loads the embedding model, so `SENTENCE_TRANSFORMERS_MODEL_NAME` must be set first. The tests set it to `all-MiniLM-L6-v2` in `test/conftest.py` unless you've exported a value.

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

| File | Covers |
|------|--------|
72 tests in total:

| File | Tests | Covers |
|------|-------|--------|
| `test/test_type_converter.py` | 27 | Serialization round trips for each supported type, nested structures, real index shapes; unregistered models and unknown type tags raise |
| `test/test_helpers.py` | 14 | `cosine_similarity` returns plain floats, chunking (including line breaks), the `base_chunk` window check, fence stripping in `parse_json` |
| `test/test_storage.py` | 10 | Model registration, `users/{user_id}/` keys, upload (no Redis write after an S3 failure), Redis-failure tolerance, cache hit, S3 fallback, S3 failure |
| `test/test_rag_parsing.py` | 9 | Valid and fenced LLM replies accepted; invalid replies raise `ValueError` |
| `test/test_config.py` | 7 | `load_config()` defaults, environment, invalid values; client factories use the config |
| `test/test_search.py` | 4 | RRF `limit`, blank queries, best-chunk metadata (stub model, no real embeddings) |
| `test/test_rag.py` | 1 | End-to-end: build → save → reload → RRF search → `RAG` with a mocked LLM |

S3 is mocked with moto (`mock_aws`). Async tests use pytest-asyncio in strict mode (`pytest.ini`). Storage tests use mock clients. The e2e test uses a real Redis at `localhost:6379` if one is running, and otherwise falls back to (mocked) S3.

The e2e test doesn't clear Redis, so a re-run within an hour loads the cached index instead of building it (`CODE_ISSUES.md`, R28). Until that's fixed, clear the test keys before a run that should rebuild:

```bash
uv run python -c "import redis; r=redis.Redis(); k=r.keys('users/test_user/*'); k and r.delete(*k)"
```
