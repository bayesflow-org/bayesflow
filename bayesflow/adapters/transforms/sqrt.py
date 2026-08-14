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

    def forward(self, data: Tensor, **kwargs) -> Tensor:
        return ops.sqrt(data)

    def inverse(self, data: Tensor, **kwargs) -> Tensor:
        return ops.square(data)

    def get_config(self) -> dict:
        return {}

    def log_det_jac(self, data: Tensor, inverse: bool = False, **kwargs) -> Tensor:
        ldj = -0.5 * ops.log(data) - ops.log(ops.convert_to_tensor(2.0, dtype=data.dtype))
        if inverse:
            ldj = -ldj
        return self._sum_except_batch(ldj)
