import numpy as np

from bayesflow.utils.serialization import serializable, serialize

from .elementwise_transform import ElementwiseTransform


@serializable("bayesflow.adapters")
class ExpandDims(ElementwiseTransform):
    """
    Expand the shape of an array.

    Parameters
    ----------
    axis : int or tuple
        The axis to expand.

    Examples
    --------
    shape (3,) array:

    >>> a = np.array([1, 2, 3])

    shape (2, 3) array:

    >>> b = np.array([[1, 2, 3], [4, 5, 6]])
    >>> dat = dict(a=a, b=b)

    >>> ed = bf.Adapter().expand_dims("a", axis=0)
    >>> new_dat = ed.forward(dat)
    >>> new_dat["a"].shape
    (1, 3)

    >>> ed = bf.Adapter().expand_dims("a", axis=1)
    >>> new_dat = ed.forward(dat)
    >>> new_dat["a"].shape
    (3, 1)

    >>> ed = bf.Adapter().expand_dims("b", axis=1)
    >>> new_dat = ed.forward(dat)
    >>> new_dat["b"].shape
    (2, 1, 3)

    It is recommended to precede this transform with a :class:`~bayesflow.adapters.transforms.ToArray` transform.
    """

    def __init__(self, *, axis: int | tuple):
        super().__init__()
        self.axis = axis

    def get_config(self) -> dict:
        return serialize({"axis": self.axis})

    def forward(self, data: np.ndarray, **kwargs) -> np.ndarray:
        return np.expand_dims(data, axis=self.axis)

    def inverse(self, data: np.ndarray, **kwargs) -> np.ndarray:
        return np.squeeze(data, axis=self.axis)
