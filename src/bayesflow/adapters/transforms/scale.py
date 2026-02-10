import numpy as np

from bayesflow.utils.serialization import serializable, serialize

from .elementwise_transform import ElementwiseTransform


@serializable("bayesflow.adapters")
class Scale(ElementwiseTransform):
    def __init__(self, scale: np.typing.ArrayLike):
        self.scale = np.array(scale)

    def get_config(self) -> dict:
        return serialize({"scale": self.scale})

    def forward(self, data: np.ndarray, **kwargs) -> np.ndarray:
        return data * self.scale

    def inverse(self, data: np.ndarray, **kwargs) -> np.ndarray:
        return data / self.scale

    def log_det_jac(self, data: np.ndarray, inverse: bool = False, **kwargs) -> np.ndarray:
        ldj = np.log(np.abs(self.scale))
        ldj = np.full(data.shape, ldj)
        if inverse:
            ldj = -ldj
        return np.sum(ldj, axis=tuple(range(1, ldj.ndim)))
