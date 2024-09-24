from collections.abc import Sequence
from keras.saving import (
    deserialize_keras_object as deserialize,
    register_keras_serializable as serializable,
    serialize_keras_object as serialize,
)

import numpy as np

from .lambda_transform import LambdaTransform


@serializable(package="bayesflow.data_adapters")
class NumpyTransform(LambdaTransform):
    """A LambdaTransform for numpy functions. Automatically serializable, unlike LambdaTransform."""

    def __init__(self, parameters: str | Sequence[str] | None = None, /, *, forward: str, inverse: str):
        self.forward_name = forward
        self.inverse_name = inverse

        forward = getattr(np, forward)
        inverse = getattr(np, inverse)
        super().__init__(parameters, forward=forward, inverse=inverse)

    @classmethod
    def from_config(cls, config: dict, custom_objects=None) -> "NumpyTransform":
        return cls(
            deserialize(config["parameters"], custom_objects),
            forward=deserialize(config["forward_name"], custom_objects),
            inverse=deserialize(config["inverse_name"], custom_objects),
        )

    def get_config(self) -> dict:
        return {
            "parameters": serialize(self.parameters),
            "forward_name": serialize(self.forward_name),
            "inverse_name": serialize(self.inverse_name),
        }
