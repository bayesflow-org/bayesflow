import numpy as np

from bayesflow.utils.serialization import serializable, serialize
from bayesflow.types import Tensor

from .elementwise_transform import ElementwiseTransform


@serializable("bayesflow.adapters")
class Shift(ElementwiseTransform):
    def __init__(self, shift: np.typing.ArrayLike):
        self.shift = np.array(shift)

    def get_config(self) -> dict:
        return serialize({"shift": self.shift})

    def _forward(self, data: np.ndarray, **kwargs) -> np.ndarray:
        return data + self.shift

    def _forward_keras(self, data: Tensor, **kwargs) -> Tensor:
        return data + self.shift

    def _inverse(self, data: np.ndarray, **kwargs) -> np.ndarray:
        return data - self.shift

    def _inverse_keras(self, data: Tensor, **kwargs) -> Tensor:
        return data - self.shift
