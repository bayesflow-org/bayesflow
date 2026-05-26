import numpy as np
import keras.ops as ops

from bayesflow.utils.serialization import serializable, serialize
from bayesflow.types import Tensor

from .elementwise_transform import ElementwiseTransform


@serializable("bayesflow.adapters")
class Scale(ElementwiseTransform):
    def __init__(self, scale: np.typing.ArrayLike):
        self.scale = np.array(scale)

    def get_config(self) -> dict:
        return serialize({"scale": self.scale})

    def _forward(self, data: np.ndarray, **kwargs) -> np.ndarray:
        return data * self.scale

    def _forward_keras(self, data: Tensor, **kwargs) -> Tensor:
        return data * self.scale

    def _inverse(self, data: np.ndarray, **kwargs) -> np.ndarray:
        return data / self.scale

    def _inverse_keras(self, data: Tensor, **kwargs) -> Tensor:
        return data / self.scale

    def _log_det_jac(self, data: np.ndarray, inverse: bool = False, **kwargs) -> np.ndarray:
        ldj = np.log(np.abs(self.scale))
        ldj = np.full(data.shape, ldj)
        if inverse:
            ldj = -ldj
        return np.sum(ldj, axis=tuple(range(1, ldj.ndim)))

    def _log_det_jac_keras(self, data: Tensor, inverse: bool = False, **kwargs) -> Tensor:
        ldj = ops.log(ops.abs(ops.convert_to_tensor(self.scale, dtype=data.dtype)))
        ldj = ops.broadcast_to(ldj, ops.shape(data))
        if inverse:
            ldj = -ldj
        return self._sum_except_batch_keras(ldj)
