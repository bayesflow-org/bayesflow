import keras.ops as ops

from bayesflow.utils.serialization import serializable, serialize
from bayesflow.types import Tensor

from .elementwise_transform import ElementwiseTransform


@serializable("bayesflow.adapters")
class Scale(ElementwiseTransform):
    def __init__(self, scale: float | Tensor):
        self.scale = ops.convert_to_tensor(scale)

    def get_config(self) -> dict:
        return serialize({"scale": self.scale})

    def forward(self, data: Tensor, **kwargs) -> Tensor:
        return data * ops.cast(self.scale, ops.dtype(data))

    def inverse(self, data: Tensor, **kwargs) -> Tensor:
        return data / ops.cast(self.scale, ops.dtype(data))

    def log_det_jac(self, data: Tensor, inverse: bool = False, **kwargs) -> Tensor:
        ldj = ops.log(ops.abs(ops.cast(self.scale, ops.dtype(data))))
        ldj = ops.broadcast_to(ldj, ops.shape(data))
        if inverse:
            ldj = -ldj
        return self._sum_except_batch(ldj)
