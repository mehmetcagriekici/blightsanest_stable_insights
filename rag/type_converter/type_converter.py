import numpy as np
from collections.abc import Callable
from collections import Counter, OrderedDict



# half dynamic class to help type conversions for msgpack
# converts existing types into list for packing
# covnerst list to existing types for unpacking
class TypeConverter:
    def __init__(self) -> None:
        # a dict to hold types to their serializers
        self.serializers: dict[str, Callable] = {}
        # dict to hold types to their deserializers
        self.deserializers: dict[str, Callable] = {}

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
                lambda d: np.array(d["value"], dtype=d["dtype"].np.reshape(d["shape"])),
                )

