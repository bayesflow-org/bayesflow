import numpy as np
from keras.saving import (
    deserialize_keras_object as deserialize,
    register_keras_serializable as serializable,
    serialize_keras_object as serialize,
)

from .elementwise_transform import ElementwiseTransform
from .transform import Transform


@serializable(package="bayesflow.data_adapters")
class MapTransform(Transform):
    """
    Implements a transform that applies a set of elementwise transforms
    to a subset of the data as given by a mapping.
    """

    def __init__(self, transform_map: dict[str, ElementwiseTransform]):
        self.transform_map = transform_map

    @classmethod
    def from_config(cls, config: dict, custom_objects=None) -> "MapTransform":
        return cls(deserialize(config["transform_map"]))

    def get_config(self) -> dict:
        return {"transform_map": serialize(self.transform_map)}

    def forward(self, data: dict[str, np.ndarray], *, strict: bool = True, **kwargs) -> dict[str, np.ndarray]:
        data = data.copy()

        required_keys = set(self.transform_map.keys())
        available_keys = set(data.keys())
        missing_keys = required_keys - available_keys

        if strict and missing_keys:
            raise KeyError(f"Missing keys: {missing_keys!r}")

        for key, transform in self.transform_map.items():
            if key in data:
                data[key] = transform.forward(data[key], **kwargs)

        return data

    def inverse(self, data: dict[str, np.ndarray], *, strict: bool = False, **kwargs) -> dict[str, np.ndarray]:
        data = data.copy()

        required_keys = set(self.transform_map.keys())
        available_keys = set(data.keys())
        missing_keys = required_keys - available_keys

        if strict and missing_keys:
            raise KeyError(f"Missing keys: {missing_keys!r}")

        for key, transform in self.transform_map.items():
            if key in data:
                data[key] = transform.inverse(data[key], **kwargs)

        return data
