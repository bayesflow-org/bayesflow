from keras.saving import (
    deserialize_keras_object as deserialize,
    register_keras_serializable as serializable,
    serialize_keras_object as serialize,
)
import numpy as np

from .elementwise_transform import ElementwiseTransform


@serializable(package="bayesflow.adapters")
class Scale(ElementwiseTransform):
    def __init__(self, scale: np.typing.ArrayLike):
        self.scale = np.array(scale)

    @classmethod
    def from_config(cls, config: dict, custom_objects=None) -> "ElementwiseTransform":
        return cls(scale=deserialize(config["scale"]))

    def get_config(self) -> dict:
        return {"scale": serialize(self.scale)}

    def forward(self, data: np.ndarray, **kwargs) -> np.ndarray:
        return data * self.scale

    def inverse(self, data: np.ndarray, **kwargs) -> np.ndarray:
        return data / self.scale

    def log_det_jac(self, data: np.ndarray, **kwargs) -> np.ndarray:
        ldj = np.log(np.abs(self.scale))
        ldj = np.full(data.shape, ldj)
        return np.sum(ldj, axis=tuple(range(1, ldj.ndim)))
