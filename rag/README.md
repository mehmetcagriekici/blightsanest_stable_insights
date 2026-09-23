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
| `semantic_index/` | `SemanticIndex`: `all-MiniLM-L6-v2` embeddings (384-dim) over sentence chunks (4 sentences, 1 overlapping). Chunk metadata: `document_id`, `chunk_index`, `total_chunks`. A document's score is its best chunk's cosine similarity; chunks resolve to documents through `docmap` by `document_id`. |
| `search/` | `HybridSearch`: loads both indexes for a user, fuses BM25 and semantic ranks with RRF (k = 60) over the union of results. Default limit 50. |
| `rag/` | `RAG`: prompt construction and JSON response parsing. The LLM function is injected via the constructor; `RAG` never selects a provider. |
| `llm/` | LLM providers with the signature `async (user_content, system_content) -> str \| None`. `ollama.py` → `llm_ollama` (development); `bedrock.py` → `llm_bedrock` (production, Converse API). |
| `storage/` | `Storage`: `upload_data(name, data)` writes to S3 then Redis (TTL 3600s); `load_data(name)` reads Redis first, falls back to S3. Redis errors are logged and never fatal. |
| `type_converter/` | `TypeConverter`: MessagePack serialization with a type registry for set, tuple, Counter, OrderedDict, defaultdict, numpy arrays, and registered Pydantic models. |
| `custom_types/` | Pydantic models: `Document`, `User`, `RagResponse` (`custom_types.py`); `DbUser`, `DbDocument` mirroring DB rows (`db_types.py`). |
| `helpers/` | Tokenizing (NLTK, English stopwords), cosine similarity, chunking, RRF score, JSON parsing. |
| `constants/` | `BM25_K1`, `BM25_B`, `SEARCH_LIMIT`. |
| `server.py` | Placeholder for the gRPC server (empty). |
| `test/` | pytest suite (see below). |

---

## Storage Layout

Each object is a MessagePack blob keyed per user, in both S3 and Redis:

```
{bucket}/{user_id}/inverted_index
{bucket}/{user_id}/docmap
{bucket}/{user_id}/term_frequencies
{bucket}/{user_id}/doc_lengths
{bucket}/{user_id}/chunk_embeddings
{bucket}/{user_id}/chunk_metadata
```

S3 is the source of truth; Redis is a disposable hot cache. The target layout uses a `users/{user_id}/` prefix, which is not implemented yet.

The bucket name and S3 credentials currently come from the `User` model (`bucket_name`, `aws_access_key_id`, `aws_secret_access_key`, `region`).

---

## Configuration

| Variable | Used by | Default |
|----------|---------|---------|
| `OLLAMA_HOST` | `llm_ollama` | `http://localhost:11434` |
| `AWS_REGION_NAME` | `llm_bedrock` | `us-east-1` |
| `BEDROCK_MODEL_ID` | `llm_bedrock` | `anthropic.claude-3-haiku-20240307-v1:0` |

Redis defaults to `localhost:6379` (constructor arguments of `Storage`). The Ollama model defaults to `gemma3`.

---

## Usage

```python
import asyncio

from custom_types.custom_types import Document, User
from llm.ollama import llm_ollama
from rag.rag import RAG
from search.hybrid_search import HybridSearch

user = User(id="user_123", aws_access_key_id="...", aws_secret_access_key="...",
            region="us-east-1", bucket_name="my-bucket")
docs = [Document(id="doc1", content="Today I felt anxious about my presentation.")]

search = HybridSearch(user, docs)
results = search.rrf_search("felt anxious")

retrieved = [Document(id=r["doc_id"], content=r["content"]) for r in results]
answer = asyncio.run(RAG(llm_ollama).rag("How did I feel?", retrieved))
print(answer.status, answer.response)
```

---

## Development

Requirements: Python 3.12 (`.python-version`). Dependencies are currently listed in the root `requirements.txt`; the `uv` + `pyproject.toml` setup described in `CLAUDE.md` is not in place yet.

Importing `helpers` downloads the NLTK `punkt_tab` and `stopwords` data on first run, so it needs network access.

Local services:

```bash
docker-compose up -d redis ollama   # from the repo root
```

### Tests

```bash
cd rag
pytest
pytest test/test_rag.py::TestRagEnd2End::test_full_pipeline -q
```

| File | Covers |
|------|--------|
| `test/test_type_converter.py` | 32 serialization round-trip tests |
| `test/test_storage.py` | 8 tests: upload, Redis-failure tolerance, cache hit, S3 fallback, S3 failure |
| `test/test_rag.py` | End-to-end: build → save → reload → RRF search → `RAG` with a mocked LLM |

S3 is mocked with moto (`mock_aws`). Async tests use pytest-asyncio in strict mode (`pytest.ini`). Storage tests mock Redis. The e2e test connects to `localhost:6379` and uses Redis if it is running; otherwise it falls back to (mocked) S3.
