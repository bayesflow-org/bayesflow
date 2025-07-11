import numpy as np

from bayesflow.utils.serialization import serializable, serialize

from .elementwise_transform import ElementwiseTransform


@serializable("bayesflow.adapters")
class Standardize(ElementwiseTransform):
    """
    Transform that when applied standardizes data using typical z-score standardization with
    fixed means and std, i.e. for some unstandardized data x the standardized version z would be

    >>> z = (x - mean(x)) / std(x)

    Important: Ensure dynamic standarization (employed by BayesFlow approximators) has been
    turned off when using this transform.

    Parameters
    ----------
    mean : int or float
        Specifies the mean (location) of the transform.
    std : int or float
        Specifies the standard deviation (scale) of the transform.

    Examples
    --------
    >>> adapter = (
            bf.adapters.Adapter().standardize(include="beta", mean=5, std=10)
    """

    def __init__(
        self,
        mean: int | float | np.ndarray,
        std: int | float | np.ndarray,
    ):
        super().__init__()

        self.mean = mean
        self.std = std

    def get_config(self) -> dict:
        config = {
            "mean": self.mean,
            "std": self.std,
        }
        return serialize(config)

    def forward(self, data: np.ndarray, **kwargs) -> np.ndarray:
        mean = np.broadcast_to(self.mean, data.shape)
        std = np.broadcast_to(self.std, data.shape)

        return (data - mean) / std

    def inverse(self, data: np.ndarray, **kwargs) -> np.ndarray:
        mean = np.broadcast_to(self.mean, data.shape)
        std = np.broadcast_to(self.std, data.shape)

        return data * std + mean

    def log_det_jac(self, data, inverse: bool = False, **kwargs) -> np.ndarray:
        std = np.broadcast_to(self.std, data.shape)
        ldj = -np.log(np.abs(std))
        if inverse:
            ldj = -ldj
        return np.sum(ldj, axis=tuple(range(1, ldj.ndim)))
