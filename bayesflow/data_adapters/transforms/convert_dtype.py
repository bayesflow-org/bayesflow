from keras.saving import (
    deserialize_keras_object as deserialize,
    register_keras_serializable as serializable,
    serialize_keras_object as serialize,
)
import numpy as np

from .elementwise_transform import ElementwiseTransform


@serializable(package="bayesflow.data_adapters")
class ConvertDType(ElementwiseTransform):
    def __init__(self, from_dtype: str, to_dtype: str):
        super().__init__()

        self.from_dtype = from_dtype
        self.to_dtype = to_dtype

    @classmethod
    def from_config(cls, config: dict, custom_objects=None) -> "ConvertDType":
        return cls(
            from_dtype=deserialize(config["from_dtype"], custom_objects),
            to_dtype=deserialize(config["to_dtype"], custom_objects),
        )

    def get_config(self) -> dict:
        return {
            "from_dtype": serialize(self.from_dtype),
            "to_dtype": serialize(self.to_dtype),
        }

    def forward(self, data: np.ndarray, **kwargs) -> np.ndarray:
        return data.astype(self.to_dtype)

    def inverse(self, data: np.ndarray, **kwargs) -> np.ndarray:
        return data.astype(self.from_dtype)
