from collections.abc import Mapping, Sequence
from keras.saving import (
    deserialize_keras_object as deserialize,
    register_keras_serializable as serializable,
    serialize_keras_object as serialize,
)
import numpy as np

from .composite_data_adapter import CompositeDataAdapter
from .data_adapter import DataAdapter
from .transforms import Transform

TRaw = Mapping[str, np.ndarray]
TProcessed = np.ndarray | None


@serializable(package="bayesflow.data_adapters")
class _ConcatenateKeysDataAdapter(DataAdapter[TRaw, TProcessed]):
    """Concatenates data from multiple keys into a single tensor."""

    def __init__(self, keys: Sequence[str]):
        if not keys:
            raise ValueError("At least one key must be provided.")

        self.keys = keys
        self.data_shapes = None
        self.is_configured = False

    def configure(self, raw_data: TRaw) -> TProcessed:
        if not self.is_configured:
            self.data_shapes = {key: value.shape for key, value in raw_data.items()}
            self.is_configured = True

        # filter and reorder data
        data = {}
        for key in self.keys:
            if key not in raw_data:
                # if a key is missing, we cannot configure, so we return None
                return None

            data[key] = raw_data[key]

        # concatenate all tensors
        return np.concatenate(list(data.values()), axis=-1)

    def deconfigure(self, processed_data: TProcessed) -> TRaw:
        if not self.is_configured:
            raise ValueError("You must call `configure` at least once before calling `deconfigure`.")

        data = {}
        start = 0
        for key in self.keys:
            stop = start + self.data_shapes[key][-1]
            data[key] = np.take(processed_data, list(range(start, stop)), axis=-1)
            start = stop

        return data

    @classmethod
    def from_config(cls, config: dict, custom_objects=None) -> "_ConcatenateKeysDataAdapter":
        instance = cls(config["keys"])
        instance.data_shapes = config.get("data_shapes")
        instance.is_configured = config.get("is_configured", False)
        return instance

    def get_config(self) -> dict:
        return {"keys": self.keys, "data_shapes": self.data_shapes, "is_configured": self.is_configured}


@serializable(package="bayesflow.data_adapters")
class ConcatenateKeysDataAdapter(CompositeDataAdapter):
    """Concatenates data from multiple keys into multiple tensors."""

    def __init__(self, *, transforms: Sequence[Transform] = None, **keys: Sequence[str]):
        self.transforms = transforms or []
        self.keys = keys
        configurators = {key: _ConcatenateKeysDataAdapter(value) for key, value in keys.items()}
        super().__init__(configurators)

    def configure(self, raw_data):
        data = raw_data

        for transform in self.transforms:
            data = transform(data, inverse=False)

        data = super().configure(data)

        return data

    def deconfigure(self, processed_data):
        data = processed_data

        data = super().deconfigure(data)

        for transform in reversed(self.transforms):
            data = transform(data, inverse=True)

        return data

    @classmethod
    def from_config(cls, config: Mapping[str, any], custom_objects=None) -> "ConcatenateKeysDataAdapter":
        return cls(**config["keys"], transforms=deserialize(config.get("transforms")))

    def get_config(self) -> dict[str, any]:
        return {"keys": self.keys, "transforms": serialize(self.transforms)}
