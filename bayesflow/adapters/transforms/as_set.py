import numpy as np
import keras.ops as ops

from bayesflow.utils.serialization import serializable
from bayesflow.types import Tensor

from .elementwise_transform import ElementwiseTransform


@serializable("bayesflow.adapters")
class AsSet(ElementwiseTransform):
    """The `.as_set(["x", "y"])` transform indicates that both `x` and `y` are treated as sets.

    That is, their values will be treated as *exchangable* such that they will imply
    the same inference regardless of the values' order.
    This is useful, for example, in a linear regression context where we can index
    the observations in arbitrary order and always get the same regression line.

    Currently, all this transform does is to ensure that the variable
    arrays are at least 3D. The 2rd dimension is treated as the
    set dimension and the 3rd dimension as the data dimension.
    In the future, the transform will have more advanced behavior
    to better ensure the correct treatment of sets.

    Examples
    --------
    >>> adapter = bf.Adapter().as_set(["x", "y"])
    """

    def _forward(self, data: np.ndarray, **kwargs) -> np.ndarray:
        return np.atleast_3d(data)

    def _forward_keras(self, data: Tensor, **kwargs) -> Tensor:
        dim = ops.ndim(data)
        if dim == 0:
            return ops.reshape(data, (1, 1, 1))
        elif dim == 1:
            shape = ops.shape(data)
            return ops.reshape(data, (1,) + shape + (1,))
        elif dim == 2:
            shape = ops.shape(data)
            return ops.reshape(data, shape + (1,))
        else:
            return data

    def _inverse(self, data: np.ndarray, **kwargs) -> np.ndarray:
        if data.shape[2] == 1:
            return np.squeeze(data, axis=2)

        return data

    def _inverse_keras(self, data: Tensor, **kwargs) -> Tensor:
        if ops.shape(data)[2] == 1:
            return ops.squeeze(data, axis=2)

        return data

    def get_config(self) -> dict:
        return {}
