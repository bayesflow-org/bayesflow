import numpy as np

from bayesflow.utils.serialization import serializable

from .elementwise_transform import ElementwiseTransform


@serializable("bayesflow.adapters")
class Sqrt(ElementwiseTransform):
    """Square-root transform a variable.

    Examples
    --------
    >>> adapter = bf.Adapter().sqrt(["x"])
    """

    def forward(self, data: np.ndarray, **kwargs) -> np.ndarray:
        return np.sqrt(data)

    def inverse(self, data: np.ndarray, **kwargs) -> np.ndarray:
        return np.square(data)

    def get_config(self) -> dict:
        return {}

    def log_det_jac(self, data: np.ndarray, inverse: bool = False, **kwargs) -> np.ndarray:
        ldj = -0.5 * np.log(data) - np.log(2)
        if inverse:
            ldj = -ldj
        return np.sum(ldj, axis=tuple(range(1, ldj.ndim)))
