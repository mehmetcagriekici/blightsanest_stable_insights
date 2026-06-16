from typing import Any
import numpy as np
from collections.abc import Callable
from collections import Counter, OrderedDict

from pydantic import BaseModel

from helpers.helpers import defaultdict_deserializer, defaultdict_serializer



# half dynamic class to help type conversions for msgpack
# converts existing types into list for packing
# covnerst list to existing types for unpacking
class TypeConverter:
    def __init__(self) -> None:
        # a dict to hold types to their serializers
        self.serializers: dict[str, Callable] = {}
        # dict to hold types to their deserializers
        self.deserializers: dict[str, Callable] = {}
        # dict to hold pydantic models
        self.pydantic_models: dict[str, type[BaseModel]]

        # register handlers
        self.register_handlers()
       
    # register types
    # register a type to their serializer and deserializer
    # handle pydantic models during the conversion recursion
    def register_types(self, type_name: str, serializer: Callable, deserializer: Callable):
        self.serializers[type_name] = serializer
        self.deserializers[type_name] = deserializer

    # register handlers both serializer and deserializer to types
    def register_handlers(self):
        # register sets
        self.register_types(
                "set",
                lambda s: list(s), # convert set to list to serialize
                lambda d: set(d), # convert list to set after deserializing
                )

        # register tuples
        self.register_types(
                "tuple",
                lambda s: list(s),
                lambda d: tuple(d),
                )
        
        # register ordered dicts
        self.register_types(
                "ordereddict",
                lambda s: list(s.items()),
                lambda d: OrderedDict(d),
                )

        # register counter to a ordinary dict to be recursively serialized
        self.register_types(
                "counter",
                lambda s: dict(s),
                lambda d: Counter(d),
                )

        # register numpy arrays
        self.register_types(
                "numpy",
                lambda arr: {
                    "dtype": str(arr.dtype),
                    "shape": arr.shape,
                    "value": arr.tolist(),
                    },
                lambda d: np.array(d["value"], dtype=d["dtype"].reshape(d["shape"])),
                )

        # register defaultdicts
        self.register_types("defaultdict", defaultdict_serializer, defaultdict_deserializer)

    # register used pydantic models for round-tripping
    def register_pydantic_models(self, model_class: type[BaseModel]):
        # metadata
        name = model_class.__name__
        self.pydantic_models[name] = model_class

        def serializer(model: BaseModel):
            return model.model_dump()

        def deserializer(data: dict):
            return model_class(**data)

        self.register_types(name, serializer, deserializer)

    # recursively convert to msgpack compatible format using the registry
    def convert_to_serializable(self, data: Any) -> Any:
        # check the existing data types one by one

        # pydantic base models
        if isinstance(data, BaseModel):
            # metadata
            name = data.__class__.__name__
            # value pydantic models initiated at the storage __init__
            content = self.serializers[name](data) if name in self.serializers else data.model_dump()
            return {"__type__": name, "value": self.convert_to_serializable(content)}
        
        # sets
        if isinstance(data, set):
            return {
                    "__type__": "set", 
                    "value": self.convert_to_serializable(self.serializers["set"](data)),
                    }

        # tuples
        if isinstance(data, tuple):
            return {
                    "__type__": "tuple",
                    "value": self.convert_to_serializable(self.serializers["tuple"](data)),
                    }

        # ordereddicts
        if isinstance(data, OrderedDict):
            return {
                    "__type__": "ordereddict",
                    "value": self.convert_to_serializable(self.serializers["ordereddict"](data)),
                    }

        # counters
        if isinstance(data, Counter):
            return {
                    "__type__": "counter",
                    "value": self.convert_to_serializable(self.serializers["counter"](data)),
                    }

        # numpy arrays
        if isinstance(data, np.ndarray):
            return {
                    "__type__": "numpy",
                    **self.serializers["numpy"](data),
                    }

        # for nested dicts and list with complex types - list with sets etc. -
        # check every element
        if isinstance(data, dict):
            return {k: self.convert_to_serializable(v) for k, v in data.items()}

        if isinstance(data, list):
            return [self.convert_to_serializable(item) for item in data]

        # if none matches, just return the data
        return data

    # restore original types
    def convert_back_from_serialized(self, data: Any) -> Any:
        # for list ensure every element is deserialized and return the final list
        if isinstance(data, list):
            return [self.convert_back_from_serialized(item) for item in data]

        # for dicts
        if isinstance(data, dict):
            # check for __type___ created during serialization
            if "__type__" in data:
                t = data["__type__"]
                v = data["value"]
                # use deserializers
                if t in self.deserializers:
                    return self.deserializers[t](v)

            return {k: self.convert_back_from_serialized(v) for k, v in data.items()}

        # for unmatched instances, just return the data
        return data
 






























