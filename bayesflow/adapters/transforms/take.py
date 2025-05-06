from collections.abc import Sequence
import numpy as np

from bayesflow.utils.serialization import serializable, serialize

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

    def forward(self, data: np.ndarray, **kwargs) -> np.ndarray:
        return np.take(data, self.indices, self.axis)

    def inverse(self, data: np.ndarray, **kwargs) -> np.ndarray:
        # not a true invertible function
        return data

    def get_config(self) -> dict:
        config = {"indices": self.indices, "axis": self.axis}

        return serialize(config)
