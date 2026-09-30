import numpy as np
import pytest

from custom_types.custom_types import Document
from semantic_index.semantic_index import SemanticIndex


# deterministic stand-in for the sentence transformer: each text maps to a
# small vector, and every encoded text is recorded
class ModelStub:
    def __init__(self, query_vector: list[float] | None = None) -> None:
        self.query_vector = query_vector
        self.encoded: list[str] = []

    def encode(self, texts):
        self.encoded.extend(texts)
        if self.query_vector is not None and len(texts) == 1:
            return np.array([self.query_vector], dtype=np.float32)
        return np.array(
            [[len(t), sum(map(ord, t)) % 97, 1.0] for t in texts], dtype=np.float32
        )


def rows_of(index: SemanticIndex, doc_id: str) -> np.ndarray:
    keep = [i for i, m in enumerate(index.chunk_metadata) if m["document_id"] == doc_id]
    return index.chunk_embeddings[keep]


# five sentences -> two chunks (4 sentences, 1 overlapping)
LONG = "One. Two. Three. Four. Five."


@pytest.fixture
def index() -> SemanticIndex:
    index = SemanticIndex(embedding_model=ModelStub())
    index.build_chunk_embeddings(
        [Document(id="doc1", content=LONG), Document(id="doc2", content="Short.")]
    )
    return index


class TestIncrementalBuild:
    def test_build_embeds_every_chunk(self, index):
        assert len(index.chunk_embeddings) == 3
        assert [m["document_id"] for m in index.chunk_metadata] == [
            "doc1",
            "doc1",
            "doc2",
        ]
        assert set(index.docmap) == {"doc1", "doc2"}

    # only the changed document is embedded again; the others keep their rows
    def test_rebuilding_a_document_replaces_only_its_chunks(self, index):
        doc2_rows = rows_of(index, "doc2").copy()
        index.model.encoded.clear()

        index.build_chunk_embeddings([Document(id="doc1", content="Changed.")])

        assert index.model.encoded == ["Changed."]
        assert len(index.chunk_embeddings) == len(index.chunk_metadata) == 2
        assert [m["document_id"] for m in index.chunk_metadata] == ["doc2", "doc1"]
        np.testing.assert_array_equal(rows_of(index, "doc2"), doc2_rows)
        assert index.docmap["doc1"].content == "Changed."

    def test_adding_a_document_appends(self, index):
        index.build_chunk_embeddings([Document(id="doc3", content="New.")])
        assert len(index.chunk_embeddings) == len(index.chunk_metadata) == 4
        assert index.chunk_metadata[-1]["document_id"] == "doc3"

    # a document whose content becomes blank keeps its docmap entry, no chunks
    def test_blank_content_has_no_chunks(self, index):
        index.build_chunk_embeddings([Document(id="doc1", content="  ")])
        assert [m["document_id"] for m in index.chunk_metadata] == ["doc2"]
        assert "doc1" in index.docmap

    def test_repeated_document_id_keeps_the_last_copy(self):
        index = SemanticIndex(embedding_model=ModelStub())
        index.build_chunk_embeddings(
            [Document(id="d", content="First."), Document(id="d", content="Second.")]
        )
        assert index.model.encoded == ["Second."]
        assert len(index.chunk_metadata) == 1

    def test_all_blank_documents_leave_the_index_empty(self):
        index = SemanticIndex(embedding_model=ModelStub())
        index.build_chunk_embeddings([Document(id="d", content="")])
        assert index.chunk_embeddings is None
        assert index.chunk_metadata == []
        assert index.search_chunks("anything") == []


class TestRemoveDocument:
    def test_removes_rows_metadata_and_docmap_entry(self, index):
        doc2_rows = rows_of(index, "doc2").copy()
        index.remove_document("doc1")
        assert [m["document_id"] for m in index.chunk_metadata] == ["doc2"]
        np.testing.assert_array_equal(index.chunk_embeddings, doc2_rows)
        assert "doc1" not in index.docmap

    def test_unknown_document_is_a_no_op(self, index):
        index.remove_document("missing")
        assert len(index.chunk_metadata) == 3


class TestParts:
    def test_export_then_restore(self, index):
        restored = SemanticIndex(index.docmap, embedding_model=ModelStub())
        restored.restore_parts(index.export_parts())
        np.testing.assert_array_equal(restored.chunk_embeddings, index.chunk_embeddings)
        assert restored.chunk_metadata == index.chunk_metadata

    # R30 regression: misaligned parts used to credit scores to the wrong doc
    def test_restore_rejects_misaligned_parts(self, index):
        parts = index.export_parts()
        parts["chunk_metadata"] = parts["chunk_metadata"][:-1]
        with pytest.raises(ValueError, match="3 rows but chunk_metadata has 2"):
            SemanticIndex(embedding_model=ModelStub()).restore_parts(parts)


# SemanticIndex with hand-built chunks and a stub model
def make_semantic_index() -> SemanticIndex:
    document = Document(id="doc1", content="text")
    index = SemanticIndex(
        {"doc1": document}, embedding_model=ModelStub(query_vector=[1.0, 0.0])
    )
    # doc1 has two chunks; the second one matches the query exactly
    index.restore_parts(
        {
            "chunk_embeddings": np.array([[0.0, 1.0], [1.0, 0.0]]),
            "chunk_metadata": [
                {"document_id": "doc1", "chunk_index": 0, "total_chunks": 2},
                {"document_id": "doc1", "chunk_index": 1, "total_chunks": 2},
            ],
        }
    )
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

    def test_empty_index_returns_empty(self):
        assert SemanticIndex(embedding_model=ModelStub()).search_chunks("q") == []
