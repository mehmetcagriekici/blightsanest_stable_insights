import math
from collections import OrderedDict

import pytest

from constants.constants import BM25_B, BM25_K1
from custom_types.custom_types import Document
from inverted_index.inverted_index import InvertedIndex
from type_converter.type_converter import TypeConverter


# in-memory storage that round-trips through the real TypeConverter, so
# save/load exercises the same serialization as S3 and Redis
class FakeStorage:
    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}
        self.converter = TypeConverter()
        self.converter.register_pydantic_models(Document)

    def upload_data(self, document_name, data):
        self.objects[document_name] = self.converter.serialize(data)

    def load_data(self, document_name):
        if document_name not in self.objects:
            return None
        return self.converter.deserialize(self.objects[document_name])


DOCUMENTS = [
    Document(id="doc1", content="cat cat cat"),
    Document(id="doc2", content="cat dog"),
    Document(id="doc3", content="dog bird fish"),
]


@pytest.fixture
def index() -> InvertedIndex:
    index = InvertedIndex(FakeStorage())
    index.build(DOCUMENTS)
    return index


class TestBuild:
    def test_fills_index_docmap_and_stats(self, index):
        assert index.get_documents("cat") == {"doc1", "doc2"}
        assert index.get_documents("dog") == {"doc2", "doc3"}
        assert index.docmap["doc1"] == DOCUMENTS[0]
        assert index.doc_lengths == {"doc1": 3, "doc2": 2, "doc3": 3}
        assert index.get_tf("doc1", "cat") == 3

    def test_drops_stopwords_and_lowercases(self):
        index = InvertedIndex(FakeStorage())
        index.build([Document(id="d", content="The Cat and the Hat")])
        assert set(index.index) == {"cat", "hat"}

    def test_unknown_token_and_document(self, index):
        assert index.get_documents("zebra") == set()
        assert index.get_tf("missing", "cat") == 0

    def test_empty_index_has_zero_average_length(self):
        assert InvertedIndex(FakeStorage()).get_avg_doc_length() == 0.0


class TestBm25Scoring:
    def test_idf(self, index):
        # n = 3 documents, "cat" appears in 2
        assert index.get_idf("cat") == pytest.approx(
            math.log((3 - 2 + 0.5) / (2 + 0.5) + 1)
        )
        # rarer terms score higher
        assert index.get_idf("bird") > index.get_idf("cat")

    def test_bm25_tf_applies_length_normalization(self, index):
        avg_len = 8 / 3
        length_norm = 1 - BM25_B + BM25_B * (3 / avg_len)
        expected = (3 * (BM25_K1 + 1)) / (3 + BM25_K1 * length_norm)
        assert index.get_bm25_tf("doc1", "cat") == pytest.approx(expected)

    def test_bm25_is_idf_times_saturated_tf(self, index):
        assert index.bm25("doc2", "dog") == pytest.approx(
            index.get_idf("dog") * index.get_bm25_tf("doc2", "dog")
        )


class TestBm25Search:
    def test_ranks_by_score(self, index):
        results = index.bm25_search("cat")
        assert isinstance(results, OrderedDict)
        # doc1 mentions "cat" three times, doc2 once, doc3 never
        assert list(results) == ["doc1", "doc2"]
        assert results["doc1"] > results["doc2"]

    def test_sums_scores_over_query_tokens(self, index):
        results = index.bm25_search("cat dog")
        assert results["doc2"] == pytest.approx(
            index.bm25("doc2", "cat") + index.bm25("doc2", "dog")
        )

    def test_respects_limit(self, index):
        assert len(index.bm25_search("cat dog", limit=1)) == 1

    @pytest.mark.parametrize("query", ["zebra", "the and", ""])
    def test_no_matching_tokens_returns_empty(self, index, query):
        assert index.bm25_search(query) == OrderedDict()


class TestPersistence:
    # a saved index loads back with identical scores
    def test_save_then_load_round_trip(self, index):
        index.save()
        assert set(index.storage.objects) == {
            "inverted_index",
            "docmap",
            "term_frequencies",
            "doc_lengths",
        }

        loaded = InvertedIndex(index.storage)
        loaded.load([])

        assert loaded.index == index.index
        assert loaded.docmap == index.docmap
        assert loaded.doc_lengths == index.doc_lengths
        assert loaded.bm25_search("cat dog") == index.bm25_search("cat dog")


# R26 regression: build() used to stack a re-added document on its old entry
class TestUpdateAndDelete:
    def test_rebuilding_a_document_replaces_it(self, index):
        index.build([Document(id="doc1", content="bird")])
        assert "doc1" not in index.get_documents("cat")
        assert index.get_tf("doc1", "cat") == 0
        assert index.get_tf("doc1", "bird") == 1
        assert index.doc_lengths["doc1"] == 1
        assert index.docmap["doc1"].content == "bird"
        assert len(index.docmap) == 3

    def test_removing_a_document(self, index):
        index.remove_document("doc2")
        assert index.get_documents("cat") == {"doc1"}
        assert index.get_documents("dog") == {"doc3"}
        assert len(index.docmap) == 2
        assert "doc2" not in index.doc_lengths
        assert "doc2" not in index.term_frequencies
        assert "doc2" not in index.bm25_search("cat dog")

    # a token no document contains any more must leave the index entirely
    def test_removing_last_holder_drops_the_token(self, index):
        index.remove_document("doc3")
        assert "bird" not in index.index
        assert index.bm25_search("bird") == OrderedDict()

    def test_removing_an_unknown_document_is_a_no_op(self, index):
        index.remove_document("missing")
        assert len(index.docmap) == 3
        assert "missing" not in index.term_frequencies

    # a loaded index (defaultdict restored from storage) supports removal too
    def test_remove_after_load(self, index):
        index.save()
        loaded = InvertedIndex(index.storage)
        loaded.load([])
        loaded.remove_document("doc1")
        assert loaded.get_documents("cat") == {"doc2"}
