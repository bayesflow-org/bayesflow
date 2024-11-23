import numpy as np

from keras.saving import (
    deserialize_keras_object as deserialize,
    serialize_keras_object as serialize,
)

from collections.abc import Sequence
from .elementwise_transform import ElementwiseTransform


class ExpandDims(ElementwiseTransform):
    """
    Expand the shape of an array.
    Examples:
        shape (3,) array:
        >>> a = np.array([1, 2, 3])
        shape (2, 3) array:
        >>> b = np.array([[1, 2, 3], [4, 5, 6]])
        >>> dat = dict(a=a, b=b)

        >>> ed = bf.adapters.transforms.ExpandDims("a", axis=0)
        >>> new_dat = ed.forward(dat)
        >>> new_dat["a"].shape
        (1, 3)

        >>> ed = bf.adapters.transforms.ExpandDims("a", axis=1)
        >>> new_dat = ed.forward(dat)
        >>> new_dat["a"].shape
        (3, 1)

        >>> ed = bf.adapters.transforms.ExpandDims("b", axis=1)
        >>> new_dat = ed.forward(dat)
        >>> new_dat["b"].shape
        (2, 1, 3)

    It is recommended to precede this transform with a :class:`bayesflow.adapters.transforms.ToArray` transform.
    """

    def __init__(self, keys: Sequence[str], *, axis: int | tuple):
        super().__init__()

        self.keys = keys
        self.axis = axis

    @classmethod
    def from_config(cls, config: dict, custom_objects=None) -> "ExpandDims":
        return cls(
            keys=deserialize(config["keys"], custom_objects),
            axis=deserialize(config["axis"], custom_objects),
        )

    def get_config(self) -> dict:
        return {
            "keys": serialize(self.keys),
            "axis": serialize(self.axis),
        }

    # noinspection PyMethodOverriding
    def forward(self, data: dict[str, any], **kwargs) -> dict[str, np.ndarray]:
        return {k: (np.expand_dims(v, axis=self.axis) if k in self.keys else v) for k, v in data.items()}

    # noinspection PyMethodOverriding
    def inverse(self, data: dict[str, any], **kwargs) -> dict[str, np.ndarray]:
        return {k: (np.squeeze(v, axis=self.axis) if k in self.keys else v) for k, v in data.items()}
