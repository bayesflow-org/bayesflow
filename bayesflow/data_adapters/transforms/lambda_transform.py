import numpy as np
from keras.saving import (
    deserialize_keras_object as deserialize,
    register_keras_serializable as serializable,
    serialize_keras_object as serialize,
)
from .elementwise_transform import ElementwiseTransform


@serializable(package="bayesflow.data_adapters")
class LambdaTransform(ElementwiseTransform):
    """
    Transforms a parameter using a pair of forward and inverse functions.

    Important note: This class is only serializable if the forward and inverse functions are serializable.
    This most likely means you will have to pass the scope that the forward and inverse functions are contained in
    to the `custom_objects` argument of the `deserialize` function when deserializing this class.
    """

    def __init__(self, *, forward: callable, inverse: callable):
        super().__init__()

        self._forward = forward
        self._inverse = inverse

    @classmethod
    def from_config(cls, config: dict, custom_objects=None) -> "LambdaTransform":
        return cls(
            forward=deserialize(config["forward"], custom_objects),
            inverse=deserialize(config["inverse"], custom_objects),
        )

    def get_config(self) -> dict:
        return {
            "forward": serialize(self._forward),
            "inverse": serialize(self._inverse),
        }

    def forward(self, data: np.ndarray, **kwargs) -> np.ndarray:
        return self._forward(data, **kwargs)

    def inverse(self, data: np.ndarray, **kwargs) -> np.ndarray:
        return self._inverse(data, **kwargs)
