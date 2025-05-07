from collections.abc import Sequence
import numpy as np

from bayesflow.utils.serialization import deserialize, serializable, serialize

from .transform import Transform


@serializable("bayesflow.adapters")
class Split(Transform):
    """This is the effective inverse of the :py:class:`~Concatenate` Transform.

    Parameters
    ----------
    key : str
        The key to split in the forward transform.
    into: Sequence[str]
        The names of each split after the forward transform.
    indices_or_sections : int | Sequence[int], optional, default: None
        The number of sections or indices to split on. If not given, will split evenly into len(into) parts.
    axis: int, optional, default: -1
        The axis to split on.
    """

    def __init__(self, key: str, into: Sequence[str], indices_or_sections: int | Sequence[int] = None, axis: int = -1):
        self.axis = axis
        self.key = key
        self.into = into

        if indices_or_sections is None:
            indices_or_sections = len(into)

        self.indices_or_sections = indices_or_sections

    @classmethod
    def from_config(cls, config: dict, custom_objects=None) -> "Split":
        return cls(**deserialize(config, custom_objects=custom_objects))

    def get_config(self) -> dict:
        config = {
            "key": self.key,
            "into": self.into,
            "indices_or_sections": self.indices_or_sections,
            "axis": self.axis,
        }
        return serialize(config)

    def forward(self, data: dict[str, np.ndarray], strict: bool = True, **kwargs) -> dict[str, np.ndarray]:
        # avoid side effects
        data = data.copy()

        if strict and self.key not in data:
            raise KeyError(self.key)
        elif self.key not in data:
            # we cannot produce a result, but also don't have to
            return data

        splits = np.split(data.pop(self.key), self.indices_or_sections, axis=self.axis)

        if len(splits) != len(self.into):
            raise ValueError(f"Requested {len(self.into)} splits, but produced {len(splits)}.")

        for key, split in zip(self.into, splits):
            data[key] = split

        return data

    def inverse(self, data: dict[str, np.ndarray], strict: bool = False, **kwargs) -> dict[str, np.ndarray]:
        # avoid side effects
        data = data.copy()

        required_keys = set(self.into)
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

        # remove each part
        splits = [data.pop(key) for key in self.into]

        # concatenate them all
        result = np.concatenate(splits, axis=self.axis)

        # store the result
        data[self.key] = result

        return data

    def extra_repr(self) -> str:
        result = "[" + ", ".join(map(repr, self.key)) + "] -> " + repr(self.into)

        if self.axis != -1:
            result += f", axis={self.axis}"

        return result
