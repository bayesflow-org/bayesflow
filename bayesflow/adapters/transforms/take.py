from collections.abc import Sequence
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

    def forward(self, data: Tensor, **kwargs) -> Tensor:
        return ops.take(data, self.indices, axis=self.axis)

    def inverse(self, data: Tensor, **kwargs) -> Tensor:
        return data

    def get_config(self) -> dict:
        config = {"indices": self.indices, "axis": self.axis}
        return serialize(config)
