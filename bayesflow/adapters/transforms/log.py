import keras.ops as ops
from bayesflow.utils.serialization import serializable, serialize
from bayesflow.types import Tensor

from .elementwise_transform import ElementwiseTransform


@serializable("bayesflow.adapters")
class Log(ElementwiseTransform):
    """Log transforms a variable.

    Parameters
    ----------
    p1 : boolean
        Add 1 to the input before taking the logarithm?

    Examples
    --------
    >>> adapter = bf.Adapter().log(["x"])
    """

    def __init__(self, *, p1: bool = False):
        super().__init__()
        self.p1 = p1

    def forward(self, data: Tensor, **kwargs) -> Tensor:
        if self.p1:
            return ops.log1p(data)
        else:
            return ops.log(data)

    def inverse(self, data: Tensor, **kwargs) -> Tensor:
        if self.p1:
            return ops.expm1(data)
        else:
            return ops.exp(data)

    def get_config(self) -> dict:
        return serialize({"p1": self.p1})

    def log_det_jac(self, data: Tensor, inverse: bool = False, **kwargs) -> Tensor:
        if self.p1:
            ldj = -ops.log1p(data)
        else:
            ldj = -ops.log(data)
        if inverse:
            ldj = -ldj
        return self._sum_except_batch(ldj)
