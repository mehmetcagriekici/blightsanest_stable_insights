import pytest

import numpy as np
from collections import Counter, OrderedDict, defaultdict

from type_converter.type_converter import TypeConverter
from custom_types.custom_types import Document


class TestTypeConverterBasicTypes:
    """Test serialization/deserialization of individual types."""

    def test_serialize_deserialize_set(self):
        """Test basic set serialization."""
        converter = TypeConverter()
        original = {1, 2, 3}
        serialized = converter.serialize(original)
        restored = converter.deserialize(serialized)
        assert restored == original
        assert isinstance(restored, set)

    def test_serialize_deserialize_empty_set(self):
        """Test empty set edge case."""
        converter = TypeConverter()
        original = set()
        serialized = converter.serialize(original)
        restored = converter.deserialize(serialized)
        assert restored == original
        assert isinstance(restored, set)

    def test_serialize_deserialize_counter(self):
        """Test Counter serialization."""
        converter = TypeConverter()
        original = Counter({"a": 5, "b": 3, "c": 1})
        serialized = converter.serialize(original)
        restored = converter.deserialize(serialized)
        assert restored == original
        assert isinstance(restored, Counter)

    def test_serialize_deserialize_empty_counter(self):
        """Test empty Counter edge case."""
        converter = TypeConverter()
        original = Counter()
        serialized = converter.serialize(original)
        restored = converter.deserialize(serialized)
        assert restored == original
        assert isinstance(restored, Counter)

    def test_serialize_deserialize_ordereddict(self):
        """Test OrderedDict serialization."""
        converter = TypeConverter()
        original = OrderedDict([("z", 1), ("y", 2), ("x", 3)])
        serialized = converter.serialize(original)
        restored = converter.deserialize(serialized)
        assert list(restored.keys()) == ["z", "y", "x"]
        assert isinstance(restored, OrderedDict)

    def test_serialize_deserialize_tuple(self):
        """Test tuple serialization."""
        converter = TypeConverter()
        original = (1, 2, 3)
        serialized = converter.serialize(original)
        restored = converter.deserialize(serialized)
        assert restored == original
        assert isinstance(restored, tuple)

    def test_serialize_deserialize_empty_tuple(self):
        """Test empty tuple edge case."""
        converter = TypeConverter()
        original = ()
        serialized = converter.serialize(original)
        restored = converter.deserialize(serialized)
        assert restored == original
        assert isinstance(restored, tuple)

    def test_serialize_deserialize_numpy_array_1d(self):
        """Test 1D numpy array serialization."""
        converter = TypeConverter()
        original = np.array([1.0, 2.0, 3.0])
        serialized = converter.serialize(original)
        restored = converter.deserialize(serialized)
        np.testing.assert_array_equal(restored, original)
        assert isinstance(restored, np.ndarray)
        assert restored.dtype == original.dtype

    def test_serialize_deserialize_numpy_array_2d(self):
        """Test 2D numpy array serialization."""
        converter = TypeConverter()
        original = np.array([[1, 2], [3, 4], [5, 6]])
        serialized = converter.serialize(original)
        restored = converter.deserialize(serialized)
        np.testing.assert_array_equal(restored, original)
        assert restored.shape == (3, 2)

    def test_serialize_deserialize_numpy_array_float32(self):
        """Test numpy array with different dtype."""
        converter = TypeConverter()
        original = np.array([1.5, 2.5, 3.5], dtype=np.float32)
        serialized = converter.serialize(original)
        restored = converter.deserialize(serialized)
        np.testing.assert_array_equal(restored, original)
        assert restored.dtype == np.float32

    def test_serialize_deserialize_defaultdict_counter(self):
        """Test defaultdict with Counter factory."""
        converter = TypeConverter()
        original = defaultdict(Counter)
        original["doc1"]["word1"] = 5
        original["doc1"]["word2"] = 3
        original["doc2"]["word1"] = 2
        
        serialized = converter.serialize(original)
        restored = converter.deserialize(serialized)
        
        assert restored["doc1"]["word1"] == 5
        assert restored["doc1"]["word2"] == 3
        assert restored["doc2"]["word1"] == 2
        assert isinstance(restored, defaultdict)

    def test_serialize_deserialize_defaultdict_dict(self):
        """Test defaultdict with dict factory."""
        converter = TypeConverter()
        original = defaultdict(dict)
        original["key1"]["nested"] = "value"
        
        serialized = converter.serialize(original)
        restored = converter.deserialize(serialized)
        
        assert restored["key1"]["nested"] == "value"
        assert isinstance(restored, defaultdict)


class TestTypeConverterNestedStructures:
    """Test serialization of nested and complex structures."""

    def test_dict_with_set_value(self):
        """Test dict containing a set."""
        converter = TypeConverter()
        original = {"my_set": {1, 2, 3}, "name": "test"}
        serialized = converter.serialize(original)
        restored = converter.deserialize(serialized)
        
        assert restored["my_set"] == {1, 2, 3}
        assert restored["name"] == "test"
        assert isinstance(restored["my_set"], set)

    def test_list_with_counter(self):
        """Test list containing Counter."""
        converter = TypeConverter()
        original = [Counter({"a": 1}), Counter({"b": 2})]
        serialized = converter.serialize(original)
        restored = converter.deserialize(serialized)
        
        assert restored[0] == Counter({"a": 1})
        assert restored[1] == Counter({"b": 2})
        assert isinstance(restored[0], Counter)

    def test_deeply_nested_structures(self):
        """Test multiple levels of nesting."""
        converter = TypeConverter()
        original = {
            "level1": {
                "level2": {
                    "my_set": {1, 2, 3},
                    "my_tuple": (4, 5, 6),
                    "my_counter": Counter({"x": 10})
                }
            }
        }
        serialized = converter.serialize(original)
        restored = converter.deserialize(serialized)
        
        assert restored["level1"]["level2"]["my_set"] == {1, 2, 3}
        assert restored["level1"]["level2"]["my_tuple"] == (4, 5, 6)
        assert restored["level1"]["level2"]["my_counter"] == Counter({"x": 10})

    def test_dict_with_numpy_and_set(self):
        """Test dict with multiple special types."""
        converter = TypeConverter()
        original = {
            "array": np.array([1, 2, 3]),
            "set": {10, 20, 30},
            "counter": Counter({"a": 5})
        }
        serialized = converter.serialize(original)
        restored = converter.deserialize(serialized)
        
        np.testing.assert_array_equal(restored["array"], original["array"])
        assert restored["set"] == {10, 20, 30}
        assert restored["counter"] == Counter({"a": 5})

    def test_list_of_tuples_with_sets(self):
        """Test list containing tuples containing sets."""
        converter = TypeConverter()
        original = [
            (1, {2, 3}),
            (4, {5, 6})
        ]
        serialized = converter.serialize(original)
        restored = converter.deserialize(serialized)
        
        assert restored[0] == (1, {2, 3})
        assert restored[1] == (4, {5, 6})
        assert isinstance(restored[0][1], set)


class TestTypeConverterEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_primitive_types_unchanged(self):
        """Test that primitives pass through correctly."""
        converter = TypeConverter()
        
        # String
        assert converter.deserialize(converter.serialize("hello")) == "hello"
        
        # Number
        assert converter.deserialize(converter.serialize(42)) == 42
        
        # Float
        assert converter.deserialize(converter.serialize(3.14)) == 3.14
        
        # Boolean
        assert converter.deserialize(converter.serialize(True)) is True
        
        # None
        assert converter.deserialize(converter.serialize(None)) is None

    def test_set_with_strings(self):
        """Test set containing strings."""
        converter = TypeConverter()
        original = {"apple", "banana", "cherry"}
        restored = converter.deserialize(converter.serialize(original))
        assert restored == original

    def test_counter_with_string_keys(self):
        """Test Counter with string keys (common use case)."""
        converter = TypeConverter()
        original = Counter({"the": 100, "and": 85, "or": 42})
        restored = converter.deserialize(converter.serialize(original))
        assert restored == original

    def test_very_large_set(self):
        """Test large set doesn't break."""
        converter = TypeConverter()
        original = set(range(10000))
        restored = converter.deserialize(converter.serialize(original))
        assert restored == original
        assert len(restored) == 10000

    def test_unicode_in_set(self):
        """Test unicode strings in set."""
        converter = TypeConverter()
        original = {"hello", "世界", "🌍"}
        restored = converter.deserialize(converter.serialize(original))
        assert restored == original

    def test_empty_nested_structures(self):
        """Test nested empty containers."""
        converter = TypeConverter()
        original = {
            "empty_set": set(),
            "empty_counter": Counter(),
            "empty_list": [],
            "empty_dict": {}
        }
        restored = converter.deserialize(converter.serialize(original))
        assert restored["empty_set"] == set()
        assert restored["empty_counter"] == Counter()
        assert restored["empty_list"] == []
        assert restored["empty_dict"] == {}


class TestTypeConverterPydanticModels:
    """Test Pydantic model serialization."""

    def test_pydantic_model_registration(self):
        """Test that Pydantic models can be registered."""
        converter = TypeConverter()
        converter.register_pydantic_models(Document)
        
        # Just verify no error on registration
        assert "Document" in converter.deserializers

    def test_serialize_deserialize_document(self):
        """Test Document Pydantic model."""
        converter = TypeConverter()
        converter.register_pydantic_models(Document)
        
        original = Document(id="doc1", content="This is a test document")
        serialized = converter.serialize(original)
        restored = converter.deserialize(serialized)
        
        assert restored.id == original.id
        assert restored.content == original.content
        assert isinstance(restored, Document)

    def test_list_of_pydantic_models(self):
        """Test list containing Pydantic models."""
        converter = TypeConverter()
        converter.register_pydantic_models(Document)
        
        original = [
            Document(id="doc1", content="content1"),
            Document(id="doc2", content="content2")
        ]
        serialized = converter.serialize(original)
        restored = converter.deserialize(serialized)
        
        assert len(restored) == 2
        assert restored[0].id == "doc1"
        assert restored[1].id == "doc2"
        assert all(isinstance(doc, Document) for doc in restored)

    def test_dict_with_pydantic_models(self):
        """Test dict with Pydantic models as values."""
        converter = TypeConverter()
        converter.register_pydantic_models(Document)
        
        original = {
            "first": Document(id="doc1", content="content1"),
            "second": Document(id="doc2", content="content2")
        }
        serialized = converter.serialize(original)
        restored = converter.deserialize(serialized)
        
        assert isinstance(restored["first"], Document)
        assert isinstance(restored["second"], Document)
        assert restored["first"].id == "doc1"
        assert restored["second"].id == "doc2"


class TestTypeConverterComplexRealWorld:
    """Test realistic RAG data structures."""

    def test_inverted_index_structure(self):
        """Test structure like InvertedIndex would produce."""
        converter = TypeConverter()
        converter.register_pydantic_models(Document)
        
        # Simulate what InvertedIndex saves
        inverted_index = {
            "the": {"doc1", "doc2", "doc3"},
            "quick": {"doc1"},
            "brown": {"doc1", "doc2"}
        }
        
        serialized = converter.serialize(inverted_index)
        restored = converter.deserialize(serialized)
        
        assert restored["the"] == {"doc1", "doc2", "doc3"}
        assert restored["quick"] == {"doc1"}
        assert isinstance(restored["the"], set)

    def test_term_frequencies_structure(self):
        """Test term_frequencies like InvertedIndex produces."""
        converter = TypeConverter()
        
        term_frequencies = defaultdict(Counter)
        term_frequencies["doc1"]["the"] = 5
        term_frequencies["doc1"]["quick"] = 2
        term_frequencies["doc2"]["the"] = 3
        
        serialized = converter.serialize(term_frequencies)
        restored = converter.deserialize(serialized)
        
        assert restored["doc1"]["the"] == 5
        assert restored["doc1"]["quick"] == 2
        assert restored["doc2"]["the"] == 3

    def test_semantic_index_chunk_metadata(self):
        """Test metadata structure like SemanticIndex produces."""
        converter = TypeConverter()
        
        chunk_metadata = [
            {"document_index": 0, "chunk_index": 0, "total_chunks": 5},
            {"document_index": 0, "chunk_index": 1, "total_chunks": 5},
            {"document_index": 1, "chunk_index": 0, "total_chunks": 3}
        ]
        
        serialized = converter.serialize(chunk_metadata)
        restored = converter.deserialize(serialized)
        
        assert len(restored) == 3
        assert restored[0]["document_index"] == 0
        assert restored[2]["total_chunks"] == 3

    def test_numpy_embeddings_array(self):
        """Test numpy arrays like SemanticIndex embeddings."""
        converter = TypeConverter()
        
        # Simulate embedding matrix (10 docs, 384 dimensions)
        embeddings = np.random.randn(10, 384).astype(np.float32)
        
        serialized = converter.serialize(embeddings)
        restored = converter.deserialize(serialized)
        
        np.testing.assert_array_almost_equal(restored, embeddings)
        assert restored.shape == (10, 384)
        assert restored.dtype == np.float32


class TestTypeConverterRoundTrip:
    """Test that serialize/deserialize preserves data integrity."""

    def test_multiple_roundtrips(self):
        """Test data survives multiple serialize/deserialize cycles."""
        converter = TypeConverter()
        
        original = {
            "set": {1, 2, 3},
            "counter": Counter({"a": 5}),
            "tuple": (4, 5, 6)
        }
        
        current = original
        for _ in range(5):
            serialized = converter.serialize(current)
            current = converter.deserialize(serialized)
        
        assert current["set"] == original["set"]
        assert current["counter"] == original["counter"]
        assert current["tuple"] == original["tuple"]
