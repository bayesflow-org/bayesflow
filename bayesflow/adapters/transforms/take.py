from collections.abc import Sequence
import numpy as np
import keras.ops as ops

from bayesflow.utils.serialization import serializable, serialize
from bayesflow.types import Tensor

from .elementwise_transform import ElementwiseTransform


@serializable(package="bayesflow.adapters")
class Take(ElementwiseTransform):
    """
    A transform to reduce the dimensionality of arrays output by the summary network
    Example: adapter.take("x", np.arange(0,3), axis=-1)
    """

    def __init__(self, indices: Sequence[int], axis: int = -1):
        super().__init__()
        self.indices = indices
        self.axis = axis

    def _forward(self, data: np.ndarray, **kwargs) -> np.ndarray:
        return np.take(data, self.indices, self.axis)

    def _forward_keras(self, data: Tensor, **kwargs) -> Tensor:
        return ops.take(data, self.indices, axis=self.axis)

    def _inverse(self, data: np.ndarray, **kwargs) -> np.ndarray:
        return data

    def _inverse_keras(self, data: Tensor, **kwargs) -> Tensor:
        return data

    def get_config(self) -> dict:
        config = {"indices": self.indices, "axis": self.axis}
        return serialize(config)
