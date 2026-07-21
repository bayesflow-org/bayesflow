import keras.ops as ops

from bayesflow.utils.serialization import serializable, serialize
from bayesflow.types import Tensor

from .elementwise_transform import ElementwiseTransform


@serializable("bayesflow.adapters")
class Standardize(ElementwiseTransform):
    """
    Transform that when applied standardizes data using typical z-score standardization with
    fixed means and std, i.e. for some unstandardized data x the standardized version z would be

    >>> z = (x - mean(x)) / std(x)

    Important: Ensure dynamic standardization (employed by BayesFlow approximators) has been
    turned off when using this transform.

    Parameters
    ----------
    mean : int or float
        Specifies the mean (location) of the transform.
    std : int or float
        Specifies the standard deviation (scale) of the transform.

    Examples
    --------
    >>> adapter = bf.Adapter().standardize(include="beta", mean=5, std=10)
    """

    def __init__(
        self,
        mean: int | float | Tensor,
        std: int | float | Tensor,
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

    def _mean_std(self, data: Tensor) -> tuple[Tensor, Tensor]:
        mean = ops.broadcast_to(ops.convert_to_tensor(self.mean, dtype=ops.dtype(data)), ops.shape(data))
        std = ops.broadcast_to(ops.convert_to_tensor(self.std, dtype=ops.dtype(data)), ops.shape(data))
        return mean, std

    def forward(self, data: Tensor, **kwargs) -> Tensor:
        mean, std = self._mean_std(data)
        return ops.divide(ops.subtract(data, mean), std)

    def inverse(self, data: Tensor, **kwargs) -> Tensor:
        mean, std = self._mean_std(data)
        return ops.add(ops.multiply(data, std), mean)

    def log_det_jac(self, data: Tensor, inverse: bool = False, **kwargs) -> Tensor:
        _, std = self._mean_std(data)
        ldj = -ops.log(ops.abs(std))
        if inverse:
            ldj = -ldj
        return self._sum_except_batch(ldj)
