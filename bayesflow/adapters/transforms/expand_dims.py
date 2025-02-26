import numpy as np

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
        self.initialize_config()

        self.keys = keys
        self.axis = axis

    # noinspection PyMethodOverriding
    def forward(self, data: dict[str, any], **kwargs) -> dict[str, np.ndarray]:
        return {k: (np.expand_dims(v, axis=self.axis) if k in self.keys else v) for k, v in data.items()}

    # noinspection PyMethodOverriding
    def inverse(self, data: dict[str, any], **kwargs) -> dict[str, np.ndarray]:
        return {k: (np.squeeze(v, axis=self.axis) if k in self.keys else v) for k, v in data.items()}
