from collections.abc import Sequence

import numpy as np
from keras.saving import (
    deserialize_keras_object as deserialize,
    register_keras_serializable as serializable,
    serialize_keras_object as serialize,
)
from .transform import ElementwiseTransform


@serializable(package="bayesflow.data_adapters")
class LambdaTransform(ElementwiseTransform):
    """
    Transforms a parameter using a pair of forward and inverse functions.

    Important note: This class is only serializable if the forward and inverse functions are serializable.
    This most likely means you will have to pass the scope that the forward and inverse functions are contained in
    to the `custom_objects` argument of the `deserialize` function when deserializing this class.
    """

    def __init__(self, parameters: str | Sequence[str] | None = None, /, *, forward: callable, inverse: callable):
        super().__init__(parameters)

        self._forward = forward
        self._inverse = inverse

    @classmethod
    def from_config(cls, config: dict, custom_objects=None) -> "LambdaTransform":
        return cls(
            deserialize(config["parameters"], custom_objects),
            forward=deserialize(config["forward"], custom_objects),
            inverse=deserialize(config["inverse"], custom_objects),
        )

    def forward(self, parameter_name: str, parameter_value: np.ndarray) -> np.ndarray:
        return self._forward(parameter_value)

    def get_config(self) -> dict:
        return {
            "parameters": serialize(self.parameters),
            "forward": serialize(self._forward),
            "inverse": serialize(self._inverse),
        }

    def inverse(self, parameter_name: str, parameter_value: np.ndarray) -> np.ndarray:
        return self._inverse(parameter_value)
