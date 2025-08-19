import numpy as np

from collections.abc import Sequence
from bayesflow.utils.serialization import serializable, serialize

from .elementwise_transform import ElementwiseTransform


@serializable("bayesflow.adapters")
class Reshape(ElementwiseTransform):

    def __init__(self, shape: int | Sequence[int]):
        super().__init__()

        if isinstance(shape, Sequence):
            shape = tuple(shape)
        self.shape = shape

    def forward(self, data: np.ndarray, **kwargs) -> np.ndarray:
        return np.reshape(data, self.shape)


    def inverse(self, data: np.ndarray, **kwargs) -> np.ndarray:
        return np.reshape(data, self.shape)


    def get_config(self) -> dict:
        return {"shape": self.shape}
