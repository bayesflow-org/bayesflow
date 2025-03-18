from keras.saving import register_keras_serializable as serializable
import numpy as np

from .elementwise_transform import ElementwiseTransform


@serializable(package="bayesflow.adapters")
class Sqrt(ElementwiseTransform):
    """Square-root transform a variable.

    Examples
    --------
    >>> adapter = bf.Adapter().sqrt(["x"])
    """

    def forward(self, data: np.ndarray, **kwargs) -> np.ndarray:
        return np.sqrt(data)

    def inverse(self, data: np.ndarray, **kwargs) -> np.ndarray:
        return np.square(data)

    @classmethod
    def from_config(cls, config: dict, custom_objects=None) -> "Sqrt":
        return cls()

    def get_config(self) -> dict:
        return {}
