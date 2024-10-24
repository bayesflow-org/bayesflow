from keras.saving import register_keras_serializable as serializable
import numpy as np


@serializable(package="bayesflow.adapters")
class Transform:
    def __call__(self, data: dict[str, np.ndarray], *, inverse: bool = False, **kwargs) -> dict[str, np.ndarray]:
        if inverse:
            return self.inverse(data, **kwargs)

        return self.forward(data, **kwargs)

    def __repr__(self):
        if e := self.extra_repr():
            return f"{self.__class__.__name__}({e})"
        return self.__class__.__name__

    @classmethod
    def from_config(cls, config: dict, custom_objects=None) -> "Transform":
        raise NotImplementedError

    def get_config(self) -> dict:
        raise NotImplementedError

    def forward(self, data: dict[str, np.ndarray], **kwargs) -> dict[str, np.ndarray]:
        raise NotImplementedError

    def inverse(self, data: dict[str, np.ndarray], **kwargs) -> dict[str, np.ndarray]:
        raise NotImplementedError

    def extra_repr(self) -> str:
        return ""
