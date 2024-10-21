import numpy as np
from keras.saving import (
    register_keras_serializable as serializable,
)

from bayesflow.utils.numpy_utils import one_hot
from .elementwise_transform import ElementwiseTransform


@serializable(package="bayesflow.data_adapters")
class OneHot(ElementwiseTransform):
    def __init__(self, num_classes: int):
        super().__init__()
        self.num_classes = num_classes

    @classmethod
    def from_config(cls, config: dict, custom_objects=None) -> "OneHot":
        return cls(num_classes=config["num_classes"])

    def get_config(self) -> dict:
        return {"num_classes": self.num_classes}

    def forward(self, data: np.ndarray, **kwargs) -> np.ndarray:
        return one_hot(data, self.num_classes)

    def inverse(self, data: np.ndarray, **kwargs) -> np.ndarray:
        return np.argmax(data, axis=-1)
