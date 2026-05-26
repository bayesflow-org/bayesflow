import numpy as np
import keras.ops as ops

from bayesflow.utils.serialization import serializable
from bayesflow.types import Tensor

from .elementwise_transform import ElementwiseTransform


@serializable("bayesflow.adapters")
class AsTimeSeries(ElementwiseTransform):
    """The `.as_time_series` transform can be used to indicate that variables shall be treated as time series.

    Currently, all this transformation does is to ensure that the variable
    arrays are at least 3D. The 2rd dimension is treated as the
    time series dimension and the 3rd dimension as the data dimension.
    In the future, the transform will have more advanced behavior
    to better ensure the correct treatment of time series data.

    Examples
    --------

    >>> adapter = bf.Adapter().as_time_series(["x", "y"])
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
