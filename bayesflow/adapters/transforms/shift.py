import numpy as np

from bayesflow.utils.serialization import serializable, serialize

from .elementwise_transform import ElementwiseTransform


@serializable("bayesflow.adapters")
class Shift(ElementwiseTransform):
    def __init__(self, shift: np.typing.ArrayLike):
        self.shift = np.array(shift)

    def get_config(self) -> dict:
        return serialize({"shift": self.shift})

    def forward(self, data: np.ndarray, **kwargs) -> np.ndarray:
        return data + self.shift

    def inverse(self, data: np.ndarray, **kwargs) -> np.ndarray:
        return data - self.shift
