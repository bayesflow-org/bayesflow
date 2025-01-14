from collections.abc import Sequence

from keras.saving import (
    deserialize_keras_object as deserialize,
    register_keras_serializable as serializable,
    serialize_keras_object as serialize,
)

from .transform import Transform


@serializable(package="bayesflow.adapters")
class Drop(Transform):
    """
    Transform to drop variables from further calculation.

    Parameters:
        keys: list of strings, containing names of data variables that should be dropped

    Example:

    >>> import bayesflow as bf
    >>> a = [1, 2, 3, 4]
    >>> b = [[1, 2], [3, 4]]
    >>> c = [[5, 6, 7, 8]]
    >>> dat = dict(a=a, b=b, c=c)
    >>> dat
        {'a': [1, 2, 3, 4], 'b': [[1, 2], [3, 4]], 'c': [[5, 6, 7, 8]]}
    >>> drop = bf.adapters.transforms.Drop(("b", "c"))
    >>> drop.forward(dat)
        {'a': [1, 2, 3, 4]}
    """

    def __init__(self, keys: Sequence[str]):
        self.keys = keys

    @classmethod
    def from_config(cls, config: dict, custom_objects=None) -> "Drop":
        return cls(keys=deserialize(config["keys"], custom_objects))

    def get_config(self) -> dict:
        return {"keys": serialize(self.keys)}

    def forward(self, data: dict[str, any], **kwargs) -> dict[str, any]:
        # no strict version because there is no requirement for the keys to be present
        return {key: value for key, value in data.items() if key not in self.keys}

    def inverse(self, data: dict[str, any], **kwargs) -> dict[str, any]:
        # non-invertible transform
        return data

    def extra_repr(self) -> str:
        return "[" + ", ".join(map(repr, self.keys)) + "]"
