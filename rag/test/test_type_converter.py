from collections import Counter, OrderedDict, defaultdict

import msgpack
import numpy as np
import pytest
from pydantic import BaseModel

from custom_types.custom_types import Document
from type_converter.type_converter import TypeConverter


def round_trip(value, converter: TypeConverter | None = None):
    converter = converter or TypeConverter()
    return converter.deserialize(converter.serialize(value))


class TestRoundTrip:
    # every registered container type comes back equal and with its own type
    @pytest.mark.parametrize(
        "original",
        [
            {1, 2, 3},
            set(),
            (1, 2, 3),
            (),
            Counter({"a": 5, "b": 3}),
            Counter(),
            OrderedDict([("z", 1), ("y", 2), ("x", 3)]),
        ],
        ids=[
            "set",
            "empty-set",
            "tuple",
            "empty-tuple",
            "counter",
            "empty-counter",
            "ordereddict",
        ],
    )
    def test_container_types(self, original):
        restored = round_trip(original)
        assert restored == original
        assert type(restored) is type(original)

    def test_ordereddict_keeps_order(self):
        restored = round_trip(OrderedDict([("z", 1), ("y", 2), ("x", 3)]))
        assert list(restored) == ["z", "y", "x"]

    @pytest.mark.parametrize("original", ["hello", 42, 3.14, True, None, [], {}])
    def test_primitives_pass_through(self, original):
        restored = round_trip(original)
        assert restored == original
        assert type(restored) is type(original)

    # numpy arrays keep their values, shape, and dtype
    @pytest.mark.parametrize(
        "original",
        [
            np.array([1.0, 2.0, 3.0]),
            np.array([[1, 2], [3, 4], [5, 6]]),
            np.array([[1.5, 2.5], [3.5, 4.5]], dtype=np.float32),
        ],
        ids=["1d-float64", "2d-int", "2d-float32"],
    )
    def test_numpy_array(self, original):
        restored = round_trip(original)
        assert isinstance(restored, np.ndarray)
        np.testing.assert_array_equal(restored, original)
        assert restored.shape == original.shape
        assert restored.dtype == original.dtype

    # special types nested inside dicts, lists, and tuples are restored at every level
    def test_nested_structures(self):
        original = {
            "level1": {
                "set": {1, 2, 3},
                "tuple": (4, 5, 6),
                "counter": Counter({"x": 10}),
                "array": np.array([1, 2, 3]),
                "empty": {"set": set(), "counter": Counter()},
            },
            "list": [(1, {2, 3}), Counter({"b": 2})],
        }
        restored = round_trip(original)

        level1 = restored["level1"]
        assert level1["set"] == {1, 2, 3}
        assert level1["tuple"] == (4, 5, 6)
        assert level1["counter"] == Counter({"x": 10})
        np.testing.assert_array_equal(level1["array"], [1, 2, 3])
        assert level1["empty"] == {"set": set(), "counter": Counter()}
        assert restored["list"] == [(1, {2, 3}), Counter({"b": 2})]
        assert isinstance(restored["list"][0][1], set)
        assert isinstance(restored["list"][1], Counter)


class TestDefaultDict:
    # the shape InvertedIndex stores as term_frequencies
    def test_counter_factory(self):
        original = defaultdict(Counter)
        original["doc1"]["word1"] = 5
        original["doc2"]["word1"] = 2

        restored = round_trip(original)

        assert restored == original
        assert isinstance(restored, defaultdict)
        assert restored.default_factory is Counter
        assert isinstance(restored["doc1"], Counter)
        # the factory still works after loading
        assert restored["unseen"]["word"] == 0

    # any factory other than Counter is restored as dict
    def test_other_factory_falls_back_to_dict(self):
        original = defaultdict(dict)
        original["key1"]["nested"] = "value"

        restored = round_trip(original)

        assert restored["key1"]["nested"] == "value"
        assert restored.default_factory is dict


class TestPydanticModels:
    @pytest.fixture
    def converter(self) -> TypeConverter:
        converter = TypeConverter()
        converter.register_pydantic_models(Document)
        return converter

    def test_single_model(self, converter):
        original = Document(id="doc1", content="This is a test document")
        restored = round_trip(original, converter)
        assert isinstance(restored, Document)
        assert restored == original

    # the shape InvertedIndex stores as docmap
    def test_models_inside_containers(self, converter):
        original = {
            "doc1": Document(id="doc1", content="content1"),
            "list": [Document(id="doc2", content="content2")],
        }
        restored = round_trip(original, converter)
        assert restored == original
        assert isinstance(restored["doc1"], Document)
        assert isinstance(restored["list"][0], Document)


class Unregistered(BaseModel):
    x: int


class TestUnregisteredTypes:
    # an unregistered model must fail loudly instead of round-tripping as a dict
    def test_serializing_unregistered_model_raises(self):
        with pytest.raises(TypeError, match="Unregistered is not registered"):
            TypeConverter().serialize(Unregistered(x=1))

    def test_deserializing_unknown_type_tag_raises(self):
        packed = msgpack.packb({"__blightsanest_type__": "Nope", "value": {"x": 1}})
        with pytest.raises(TypeError, match="no deserializer registered for 'Nope'"):
            TypeConverter().deserialize(packed)
