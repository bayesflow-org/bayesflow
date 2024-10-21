from collections.abc import Sequence

import numpy as np
from keras.saving import (
    deserialize_keras_object as deserialize,
    register_keras_serializable as serializable,
    serialize_keras_object as serialize,
)

from .transform import Transform


@serializable(package="bayesflow.data_adapters")
class Concatenate(Transform):
    """Concatenate multiple arrays into a new key."""

    def __init__(self, keys: Sequence[str], *, into: str, axis: int = -1):
        self.keys = keys
        self.into = into
        self.axis = axis

        self.indices = None

    @classmethod
    def from_config(cls, config: dict, custom_objects=None) -> "Concatenate":
        return cls(
            keys=deserialize(config["keys"], custom_objects),
            into=deserialize(config["into"], custom_objects),
            axis=deserialize(config["axis"], custom_objects),
        )

    def get_config(self) -> dict:
        return {
            "keys": serialize(self.keys),
            "into": serialize(self.into),
            "axis": serialize(self.axis),
        }

    def forward(self, data: dict[str, any], *, strict: bool = True, **kwargs) -> dict[str, any]:
        if not strict and self.indices is None:
            raise ValueError("Cannot call `forward` with `strict=False` before calling `forward` with `strict=True`.")

        # copy to avoid side effects
        data = data.copy()

        required_keys = set(self.keys)
        available_keys = set(data.keys())
        common_keys = available_keys & required_keys
        missing_keys = required_keys - available_keys

        if strict and missing_keys:
            # invalid call
            raise KeyError(f"Missing keys: {missing_keys!r}")
        elif missing_keys:
            # we cannot produce a result, but should still remove the keys
            for key in common_keys:
                data.pop(key)

            return data

        if self.indices is None:
            # remember the indices of the parts in the concatenated array
            self.indices = np.cumsum([data[key].shape[self.axis] for key in self.keys]).tolist()

        # remove each part
        parts = [data.pop(key) for key in self.keys]

        # concatenate them all
        result = np.concatenate(parts, axis=self.axis)

        # store the result
        data[self.into] = result

        return data

    def inverse(self, data: dict[str, any], *, strict: bool = False, **kwargs) -> dict[str, any]:
        if self.indices is None:
            raise RuntimeError("Cannot call `inverse` before calling `forward` at least once.")

        # copy to avoid side effects
        data = data.copy()

        if strict and self.into not in data:
            # invalid call
            raise KeyError(f"Missing key: {self.into!r}")
        elif self.into not in data:
            # nothing to do
            return data

        # split the concatenated array and remove the concatenated key
        keys = self.keys
        values = np.split(data.pop(self.into), self.indices, axis=self.axis)

        # restore the parts
        data |= dict(zip(keys, values))

        return data
