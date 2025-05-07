import numpy as np

from bayesflow.utils.serialization import serializable, serialize

from .elementwise_transform import ElementwiseTransform


@serializable("bayesflow.adapters")
class Log(ElementwiseTransform):
    """Log transforms a variable.

    Parameters
    ----------
    p1 : boolean
        Add 1 to the input before taking the logarithm?

    Examples
    --------
    >>> adapter = bf.Adapter().log(["x"])
    """

    def __init__(self, *, p1: bool = False):
        super().__init__()
        self.p1 = p1

    def forward(self, data: np.ndarray, **kwargs) -> np.ndarray:
        if self.p1:
            return np.log1p(data)
        else:
            return np.log(data)

    def inverse(self, data: np.ndarray, **kwargs) -> np.ndarray:
        if self.p1:
            return np.expm1(data)
        else:
            return np.exp(data)

    def get_config(self) -> dict:
        return serialize({"p1": self.p1})

    def log_det_jac(self, data: np.ndarray, inverse: bool = False, **kwargs) -> np.ndarray:
        if self.p1:
            ldj = -np.log1p(data)
        else:
            ldj = -np.log(data)
        if inverse:
            ldj = -ldj
        return np.sum(ldj, axis=tuple(range(1, ldj.ndim)))
