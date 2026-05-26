import numpy as np
import keras.ops as ops

from bayesflow.utils.serialization import serializable
from bayesflow.types import Tensor

from .elementwise_transform import ElementwiseTransform


@serializable("bayesflow.adapters")
class Sqrt(ElementwiseTransform):
    """Square-root transform a variable.

    Examples
    --------
    >>> adapter = bf.Adapter().sqrt(["x"])
    """

    def _forward(self, data: np.ndarray, **kwargs) -> np.ndarray:
        return np.sqrt(data)

    def _forward_keras(self, data: Tensor, **kwargs) -> Tensor:
        return ops.sqrt(data)

    def _inverse(self, data: np.ndarray, **kwargs) -> np.ndarray:
        return np.square(data)

    def _inverse_keras(self, data: Tensor, **kwargs) -> Tensor:
        return ops.square(data)

    def get_config(self) -> dict:
        return {}

    def _log_det_jac(self, data: np.ndarray, inverse: bool = False, **kwargs) -> np.ndarray:
        ldj = -0.5 * np.log(data) - np.log(2)
        if inverse:
            ldj = -ldj
        return np.sum(ldj, axis=tuple(range(1, ldj.ndim)))

    def _log_det_jac_keras(self, data: Tensor, inverse: bool = False, **kwargs) -> Tensor:
        ldj = -0.5 * ops.log(data) - ops.log(ops.convert_to_tensor(2.0, dtype=data.dtype))
        if inverse:
            ldj = -ldj
        return self._sum_except_batch_keras(ldj)
