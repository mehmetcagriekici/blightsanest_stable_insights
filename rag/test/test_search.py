import numpy as np

from search.hybrid_search import HybridSearch
from semantic_index.semantic_index import SemanticIndex


# HybridSearch with stubbed sub-searches: no model, storage, or indexes needed
def make_hybrid(bm25: dict[str, float], semantic: list[dict]) -> HybridSearch:
    search = HybridSearch.__new__(HybridSearch)  # skip __init__
    search.bm25_search = lambda query, limit: bm25
    search.semantic_search = lambda query, limit: semantic
    search.inverted_index = type("InvertedIndexStub", (), {"docmap": {}})()
    return search


class TestRrfSearch:
    # the union of both result sets can hold up to 2 x limit documents
    def test_respects_limit(self):
        search = make_hybrid(
            {"a": 3.0, "b": 2.0},
            [
                {"id": "c", "score": 0.9, "content": "c"},
                {"id": "d", "score": 0.8, "content": "d"},
            ],
        )
        assert len(search.rrf_search("q", limit=2)) == 2
        assert len(search.rrf_search("q", limit=10)) == 4

    def test_blank_query_returns_empty(self):
        search = make_hybrid({"a": 1.0}, [{"id": "a", "score": 1.0, "content": "a"}])
        assert search.rrf_search("") == []
        assert search.rrf_search("   ") == []


# SemanticIndex with hand-built chunks and a stub model
class ModelStub:
    def __init__(self, query_vector: list[float]) -> None:
        self.query_vector = np.array(query_vector)

    def encode(self, texts):
        return [self.query_vector]


def make_semantic_index() -> SemanticIndex:
    document = type("Doc", (), {"id": "doc1", "content": "text"})()
    index = SemanticIndex.__new__(SemanticIndex)  # skip __init__
    index.model = ModelStub([1.0, 0.0])
    # doc1 has two chunks; the second one matches the query exactly
    index.chunk_embeddings = np.array([[0.0, 1.0], [1.0, 0.0]])
    index.chunk_metadata = [
        {"document_id": "doc1", "chunk_index": 0, "total_chunks": 2},
        {"document_id": "doc1", "chunk_index": 1, "total_chunks": 2},
    ]
    index.documents = [document]
    index.docmap = {"doc1": document}
    return index


class TestSearchChunks:
    # metadata must describe the chunk that produced the document's score
    def test_metadata_is_from_best_chunk(self):
        results = make_semantic_index().search_chunks("q")
        assert len(results) == 1
        assert results[0]["metadata"]["chunk_index"] == 1
        assert results[0]["score"] == 1.0

    def test_blank_query_returns_empty(self):
        assert make_semantic_index().search_chunks("  ") == []
