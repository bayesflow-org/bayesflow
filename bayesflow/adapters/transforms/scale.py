import keras.ops as ops

from bayesflow.utils.serialization import serializable, serialize
from bayesflow.types import Tensor

from .elementwise_transform import ElementwiseTransform


@serializable("bayesflow.adapters")
class Scale(ElementwiseTransform):
    def __init__(self, scale: float | Tensor):
        self.scale = scale

    def get_config(self) -> dict:
        return serialize({"scale": self.scale})

    def forward(self, data: Tensor, **kwargs) -> Tensor:
        scale = ops.convert_to_tensor(self.scale, dtype=ops.dtype(data))
        return ops.multiply(data, scale)

    def inverse(self, data: Tensor, **kwargs) -> Tensor:
        scale = ops.convert_to_tensor(self.scale, dtype=ops.dtype(data))
        return ops.divide(data, scale)

    def log_det_jac(self, data: Tensor, inverse: bool = False, **kwargs) -> Tensor:
        scale = ops.convert_to_tensor(self.scale, dtype=ops.dtype(data))
        ldj = ops.log(ops.abs(scale))
        ldj = ops.broadcast_to(ldj, ops.shape(data))
        if inverse:
            ldj = -ldj
        return self._sum_except_batch(ldj)
