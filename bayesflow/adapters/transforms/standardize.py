import numpy as np
import keras.ops as ops

from bayesflow.utils.serialization import serializable, serialize
from bayesflow.types import Tensor

from .elementwise_transform import ElementwiseTransform


@serializable("bayesflow.adapters")
class Standardize(ElementwiseTransform):
    """
    Transform that when applied standardizes data using typical z-score standardization with
    fixed means and std, i.e. for some unstandardized data x the standardized version z would be

    >>> z = (x - mean(x)) / std(x)

    Important: Ensure dynamic standardization (employed by BayesFlow approximators) has been
    turned off when using this transform.

    Parameters
    ----------
    mean : int or float
        Specifies the mean (location) of the transform.
    std : int or float
        Specifies the standard deviation (scale) of the transform.

    Examples
    --------
    >>> adapter = bf.Adapter().standardize(include="beta", mean=5, std=10)
    """

    def __init__(
        self,
        mean: int | float | np.ndarray,
        std: int | float | np.ndarray,
    ):
        super().__init__()

        self.mean = mean
        self.std = std

    def get_config(self) -> dict:
        config = {
            "mean": self.mean,
            "std": self.std,
        }
        return serialize(config)

    def _forward(self, data: np.ndarray, **kwargs) -> np.ndarray:
        mean = np.broadcast_to(self.mean, data.shape)
        std = np.broadcast_to(self.std, data.shape)
        return (data - mean) / std

    def _forward_keras(self, data: Tensor, **kwargs) -> Tensor:
        mean = ops.broadcast_to(ops.convert_to_tensor(self.mean, dtype=data.dtype), ops.shape(data))
        std = ops.broadcast_to(ops.convert_to_tensor(self.std, dtype=data.dtype), ops.shape(data))
        return (data - mean) / std

    def _inverse(self, data: np.ndarray, **kwargs) -> np.ndarray:
        mean = np.broadcast_to(self.mean, data.shape)
        std = np.broadcast_to(self.std, data.shape)
        return data * std + mean

    def _inverse_keras(self, data: Tensor, **kwargs) -> Tensor:
        mean = ops.broadcast_to(ops.convert_to_tensor(self.mean, dtype=data.dtype), ops.shape(data))
        std = ops.broadcast_to(ops.convert_to_tensor(self.std, dtype=data.dtype), ops.shape(data))
        return data * std + mean

    def _log_det_jac(self, data: np.ndarray, inverse: bool = False, **kwargs) -> np.ndarray:
        std = np.broadcast_to(self.std, data.shape)
        ldj = -np.log(np.abs(std))
        if inverse:
            ldj = -ldj
        return np.sum(ldj, axis=tuple(range(1, ldj.ndim)))

    def _log_det_jac_keras(self, data: Tensor, inverse: bool = False, **kwargs) -> Tensor:
        std = ops.broadcast_to(ops.convert_to_tensor(self.std, dtype=data.dtype), ops.shape(data))
        ldj = -ops.log(ops.abs(std))
        if inverse:
            ldj = -ldj
        return self._sum_except_batch_keras(ldj)
