from keras.saving import register_keras_serializable as serializable
import numpy as np


@serializable(package="bayesflow.adapters")
class ElementwiseTransform:
    """Base class on which other transforms are based"""

    def __call__(self, data: np.ndarray, inverse: bool = False, **kwargs) -> np.ndarray:
        if inverse:
            return self.inverse(data, **kwargs)

        return self.forward(data, **kwargs)

    @classmethod
    def from_config(cls, config: dict, custom_objects=None) -> "ElementwiseTransform":
        raise NotImplementedError

    def get_config(self) -> dict:
        raise NotImplementedError

    def forward(self, data: np.ndarray, **kwargs) -> np.ndarray:
        raise NotImplementedError

    def inverse(self, data: np.ndarray, **kwargs) -> np.ndarray:
        raise NotImplementedError
