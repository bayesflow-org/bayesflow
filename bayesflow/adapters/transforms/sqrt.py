import numpy as np

from bayesflow.utils.serialization import serializable

from .elementwise_transform import ElementwiseTransform


@serializable
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

    def get_config(self) -> dict:
        return {}
