import numpy as np

from bayesflow.utils.serialization import serializable, serialize

from .elementwise_transform import ElementwiseTransform


@serializable
class Scale(ElementwiseTransform):
    def __init__(self, scale: np.typing.ArrayLike):
        self.scale = np.array(scale)

    def get_config(self) -> dict:
        return serialize({"scale": self.scale})

    def forward(self, data: np.ndarray, **kwargs) -> np.ndarray:
        return data * self.scale

    def inverse(self, data: np.ndarray, **kwargs) -> np.ndarray:
        return data / self.scale
