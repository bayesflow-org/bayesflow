import keras.ops as ops

from bayesflow.utils.serialization import serializable, serialize
from bayesflow.types import Tensor

from .elementwise_transform import ElementwiseTransform


@serializable("bayesflow.adapters")
class Shift(ElementwiseTransform):
    def __init__(self, shift: float | Tensor):
        self.shift = shift

    def get_config(self) -> dict:
        return serialize({"shift": self.shift})

    def forward(self, data: Tensor, **kwargs) -> Tensor:
        shift = ops.convert_to_tensor(self.shift, dtype=ops.dtype(data))
        return ops.add(data, shift)

    def inverse(self, data: Tensor, **kwargs) -> Tensor:
        shift = ops.convert_to_tensor(self.shift, dtype=ops.dtype(data))
        return ops.subtract(data, shift)
