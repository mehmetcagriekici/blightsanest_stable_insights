import pytest
import json
from moto import mock_aws
import boto3

from rag.rag import RAG
from search.hybrid_search import HybridSearch

class TestRagEnd2End:
    """Test entire RAG system working together"""

    @mock_aws
    @pytest.mark.asyncio
    async def test_full_pipeline(self, mock_user, mock_documents, redis_connection):
        """
        Full e2e flow:
        1. Create indexes with documents
        2. Save to S3 + Redis
        3. Load fresh indexes from storage
        4. Run search
        5. Run RAG with mock LLM
        6. Assert response
        """
        
        # Setup S3
        s3 = boto3.client("s3", region_name="us-east-1")
        s3.create_bucket(Bucket="test_bucket")
        
        # --- PHASE 1: Build and save indexes ---
        search = HybridSearch(mock_user, mock_documents)
        
        # Verify indexes were built
        assert len(search.inverted_index.index) > 0
        assert search.semantic_index.chunk_embeddings is not None
        
        # --- PHASE 2: Load fresh instances (simulate new request) ---
        search_reloaded = HybridSearch(mock_user, mock_documents)
        
        # Verify indexes loaded from storage
        assert len(search_reloaded.inverted_index.index) > 0
        assert search_reloaded.semantic_index.chunk_embeddings is not None
        
        # --- PHASE 3: Search ---
        results = search_reloaded.rrf_search("felt anxious")
        assert len(results) > 0
        assert results[0]["doc_id"] == "doc1"
        
        # --- PHASE 4: RAG ---
        async def mock_generate(user_prompt: str, system_prompt: str) -> str:
            return json.dumps({
                "status": "found",
                "response": "You felt anxious about your presentation."
            })
    
        rag = RAG(generate=mock_generate)
        rag_results = await rag.rag("What made me anxious?", results)
        
        # --- PHASE 5: Assert ---
        assert rag_results.status == "found"
        assert rag_results.response == "You felt anxious about your presentation."
