import numpy as np

from collections.abc import Sequence
from bayesflow.utils.serialization import serializable, serialize

from .elementwise_transform import ElementwiseTransform


@serializable("bayesflow.adapters")
class Squeeze(ElementwiseTransform):
    """
    Squeeze dimensions of an array.

    Parameters
    ----------
    axis : int or tuple
        The axis to squeeze. As the number of batch dimensions might change, we advise using negative
        numbers (i.e., indexing from the end instead of the start).

    Examples
    --------
    shape (3, 1) array:

    >>> a = np.array([[1], [2], [3]])

    >>> sq = bf.adapters.transforms.Squeeze(axis=-1)
    >>> sq.forward(a).shape
    (3,)

    It is recommended to precede this transform with a :class:`~bayesflow.adapters.transforms.ToArray` transform.
    """

    def __init__(self, *, axis: int | Sequence[int]):
        super().__init__()
        if isinstance(axis, Sequence):
            axis = tuple(axis)
        self.axis = axis

    def get_config(self) -> dict:
        return serialize({"axis": self.axis})

    def forward(self, data: np.ndarray, **kwargs) -> np.ndarray:
        return np.squeeze(data, axis=self.axis)

    def inverse(self, data: np.ndarray, **kwargs) -> np.ndarray:
        return np.expand_dims(data, axis=self.axis)
