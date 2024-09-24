from collections.abc import Mapping, Sequence
from keras.saving import (
    deserialize_keras_object as deserialize,
    register_keras_serializable as serializable,
    serialize_keras_object as serialize,
)
import numpy as np

from .transform import ElementwiseTransform


@serializable(package="bayesflow.data_adapters")
class Standardize(ElementwiseTransform):
    """Normalizes a parameter to have zero mean and unit standard deviation.
    By default, this is lazily initialized; the mean and standard deviation are computed from the first batch of data.
    For eager initialization, pass the mean and standard deviation to the constructor.
    """

    def __init__(
        self,
        parameters: str | Sequence[str] | None = None,
        /,
        *,
        means: Mapping[str, np.ndarray] = None,
        stds: Mapping[str, np.ndarray] = None,
    ):
        super().__init__(parameters)
        self.means = means or {}
        self.stds = stds or {}

    @classmethod
    def from_config(cls, config: dict, custom_objects=None) -> "Standardize":
        return cls(
            deserialize(config["parameters"], custom_objects),
            means=deserialize(config["means"], custom_objects),
            stds=deserialize(config["stds"], custom_objects),
        )

    def forward(self, parameter_name: str, parameter_value: np.ndarray) -> np.ndarray:
        if parameter_name not in self.means:
            self.means[parameter_name] = np.mean(
                parameter_value, axis=tuple(range(parameter_value.ndim)), keepdims=True
            )
        if parameter_name not in self.stds:
            self.stds[parameter_name] = np.std(parameter_value, axis=tuple(range(parameter_value.ndim)), keepdims=True)

        return (parameter_value - self.means[parameter_name]) / self.stds[parameter_name]

    def get_config(self) -> dict:
        return {
            "parameters": serialize(self.parameters),
            "means": serialize(self.means),
            "stds": serialize(self.stds),
        }

    def inverse(self, parameter_name: str, parameter_value: np.ndarray) -> np.ndarray:
        if not self.means or not self.stds:
            raise ValueError(
                f"Cannot call `inverse` before calling `forward` at least once for parameter {parameter_name}."
            )

        return parameter_value * self.stds[parameter_name] + self.means[parameter_name]
