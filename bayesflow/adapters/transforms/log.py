import numpy as np

from keras.saving import (
    deserialize_keras_object as deserialize,
    serialize_keras_object as serialize,
)

from .elementwise_transform import ElementwiseTransform


class Log(ElementwiseTransform):
    """Log transforms a variable.

    Parameters
    ----------
    p1 : boolean
        Add 1 to the input before taking the logarithm?

    Examples
    --------
    >>> adapter = bf.Adapter().log(["x"])
    """

    def __init__(self, *, p1: bool = False):
        super().__init__()
        self.p1 = p1

    def forward(self, data: np.ndarray, **kwargs) -> np.ndarray:
        if self.p1:
            return np.log1p(data)
        else:
            return np.log(data)

    def inverse(self, data: np.ndarray, **kwargs) -> np.ndarray:
        if self.p1:
            return np.expm1(data)
        else:
            return np.exp(data)

    @classmethod
    def from_config(cls, config: dict, custom_objects=None) -> "Log":
        return cls(
            p1=deserialize(config["p1"], custom_objects),
        )

    def get_config(self) -> dict:
        return {
            "p1": serialize(self.p1),
        }
