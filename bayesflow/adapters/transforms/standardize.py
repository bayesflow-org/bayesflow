from keras.saving import (
    deserialize_keras_object as deserialize,
    register_keras_serializable as serializable,
    serialize_keras_object as serialize,
)
import numpy as np

from .elementwise_transform import ElementwiseTransform


@serializable(package="bayesflow.adapters")
class Standardize(ElementwiseTransform):
    def __init__(
        self,
        mean: int | float | np.ndarray = None,
        std: int | float | np.ndarray = None,
        axis: int = None,
        momentum: float | None = 0.99,
    ):
        super().__init__()

        self.mean = mean
        self.std = std
        self.axis = axis
        self.momentum = momentum

    @classmethod
    def from_config(cls, config: dict, custom_objects=None) -> "Standardize":
        return cls(
            mean=deserialize(config["mean"], custom_objects),
            std=deserialize(config["std"], custom_objects),
            axis=deserialize(config["axis"], custom_objects),
            momentum=deserialize(config["momentum"], custom_objects),
        )

    def get_config(self) -> dict:
        return {
            "mean": serialize(self.mean),
            "std": serialize(self.std),
            "axis": serialize(self.axis),
            "momentum": serialize(self.momentum),
        }

    def forward(self, data: np.ndarray, stage: str = "training", **kwargs) -> np.ndarray:
        if self.axis is None:
            self.axis = tuple(range(data.ndim - 1))

        if self.mean is None:
            self.mean = np.mean(data, axis=self.axis, keepdims=True)
        else:
            if self.momentum is not None and stage == "training":
                self.mean = self.momentum * self.mean + (1.0 - self.momentum) * np.mean(
                    data, axis=self.axis, keepdims=True
                )

        if self.std is None:
            self.std = np.std(data, axis=self.axis, keepdims=True, ddof=1)
        else:
            if self.momentum is not None and stage == "training":
                self.std = self.momentum * self.std + (1.0 - self.momentum) * np.std(
                    data, axis=self.axis, keepdims=True, ddof=1
                )

        mean = np.broadcast_to(self.mean, data.shape)
        std = np.broadcast_to(self.std, data.shape)

        return (data - mean) / std

    def inverse(self, data: np.ndarray, **kwargs) -> np.ndarray:
        if self.mean is None or self.std is None:
            raise RuntimeError("Cannot call `inverse` before calling `forward` at least once.")

        mean = np.broadcast_to(self.mean, data.shape)
        std = np.broadcast_to(self.std, data.shape)

        return data * std + mean
