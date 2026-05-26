import numpy as np
import keras.ops as ops

from bayesflow.utils.serialization import serializable, serialize
from bayesflow.utils.numpy_utils import one_hot
from bayesflow.types import Tensor

from .elementwise_transform import ElementwiseTransform


@serializable("bayesflow.adapters")
class OneHot(ElementwiseTransform):
    """
    Changes data to be one-hot encoded.

    Parameters
    ----------
    num_classes : int
        Number of classes for the encoding.
    """

    def __init__(self, num_classes: int):
        super().__init__()
        self.num_classes = num_classes

    def get_config(self) -> dict:
        return serialize({"num_classes": self.num_classes})

    def _forward(self, data: np.ndarray, **kwargs) -> np.ndarray:
        return one_hot(data, self.num_classes)

    def _forward_keras(self, data: Tensor, **kwargs) -> Tensor:
        return ops.one_hot(data, self.num_classes)

    def _inverse(self, data: np.ndarray, **kwargs) -> np.ndarray:
        return np.argmax(data, axis=-1)

    def _inverse_keras(self, data: Tensor, **kwargs) -> Tensor:
        return ops.argmax(data, axis=-1)
