import json
from uuid import uuid4

import pytest
import redis
from moto import mock_aws
from redis.exceptions import RedisError

from config.config import Config
from custom_types.custom_types import Document, User
from rag.rag import RAG
from search.hybrid_search import HybridSearch
from storage.clients import create_redis_client, create_s3_client
from storage.storage import Storage


@pytest.fixture
def config() -> Config:
    return Config(bucket_name="test_bucket", region="us-east-1")


@pytest.fixture
def redis_connection(config) -> redis.Redis:
    return create_redis_client(config)


# S3 is a fresh moto mock per test, but Redis may be a real, shared instance.
# A unique user per run means no cached keys from an earlier run can match,
# so phase 1 really builds; the run's own keys are removed afterwards
@pytest.fixture
def e2e_user(redis_connection):
    user = User(id=f"e2e-{uuid4().hex}")
    yield user
    try:
        keys = list(redis_connection.scan_iter(match=f"users/{user.id}/*"))
        if keys:
            redis_connection.delete(*keys)
    except RedisError:
        # redis is optional; if it is down nothing was cached
        pass


class TestRagEnd2End:
    """Test entire RAG system working together"""

    @pytest.mark.asyncio
    async def test_full_pipeline(
        self, config, redis_connection, e2e_user, mock_documents
    ):
        """
        Full e2e flow:
        1. Create indexes with documents
        2. Save to S3 + Redis
        3. Load fresh indexes from storage
        4. Run search
        5. Run RAG with mock LLM
        6. Assert response
        """
        # use mock_aws as a context manager (not a decorator): a decorator
        # wraps the coroutine in a sync function, which stops pytest-asyncio
        # from awaiting the test
        with mock_aws():
            # clients are created once and shared, as the service will do
            s3 = create_s3_client(config)
            s3.create_bucket(Bucket=config.bucket_name)

            # --- PHASE 1: Build and save indexes ---
            storage = Storage(e2e_user, config.bucket_name, s3, redis_connection)
            search = HybridSearch(storage, mock_documents)

            # Verify indexes were built and saved to S3, the source of truth
            assert len(search.inverted_index.index) > 0
            assert search.semantic_index.chunk_embeddings is not None
            saved = s3.list_objects_v2(
                Bucket=config.bucket_name, Prefix=f"users/{e2e_user.id}/"
            )
            assert {obj["Key"].rsplit("/", 1)[1] for obj in saved["Contents"]} == {
                "inverted_index",
                "docmap",
                "term_frequencies",
                "doc_lengths",
                "chunk_embeddings",
                "chunk_metadata",
            }

            # --- PHASE 2: Load fresh instances (simulate new request) ---
            storage_reloaded = Storage(
                e2e_user, config.bucket_name, s3, redis_connection
            )
            search_reloaded = HybridSearch(storage_reloaded, mock_documents)

            # Verify indexes loaded from storage
            assert len(search_reloaded.inverted_index.index) > 0
            assert search_reloaded.semantic_index.chunk_embeddings is not None

            # --- PHASE 3: Search ---
            results = search_reloaded.rrf_search("felt anxious")
            assert len(results) > 0
            assert results[0]["doc_id"] == "doc1"

            # convert ranked search results into Documents for the RAG step
            # (this is the boundary the API orchestrator will own)
            retrieved_documents = [
                Document(id=result["doc_id"], content=result["content"])
                for result in results
            ]

            # --- PHASE 4: RAG ---
            async def mock_generate(user_prompt: str, system_prompt: str) -> str:
                return json.dumps(
                    {
                        "status": "found",
                        "response": "You felt anxious about your presentation.",
                    }
                )

            rag = RAG(generate=mock_generate)
            rag_results = await rag.rag("What made me anxious?", retrieved_documents)

            # --- PHASE 5: Assert ---
            assert rag_results.status == "found"
            assert rag_results.response == "You felt anxious about your presentation."
