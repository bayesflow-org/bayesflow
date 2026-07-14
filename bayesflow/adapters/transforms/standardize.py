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

        self.mean = ops.convert_to_tensor(mean)
        self.std = ops.convert_to_tensor(std)

    def get_config(self) -> dict:
        config = {
            "mean": self.mean,
            "std": self.std,
        }
        return serialize(config)

    def forward(self, data: Tensor, **kwargs) -> Tensor:
        mean = ops.broadcast_to(ops.cast(self.mean, dtype=ops.dtype(data)), ops.shape(data))
        std = ops.broadcast_to(ops.cast(self.std, dtype=ops.dtype(data)), ops.shape(data))
        return (data - mean) / std

    def inverse(self, data: Tensor, **kwargs) -> Tensor:
        mean = ops.broadcast_to(ops.cast(self.mean, dtype=ops.dtype(data)), ops.shape(data))
        std = ops.broadcast_to(ops.cast(self.std, dtype=ops.dtype(data)), ops.shape(data))
        return data * std + mean

    def log_det_jac(self, data: Tensor, inverse: bool = False, **kwargs) -> Tensor:
        std = ops.broadcast_to(ops.cast(self.std, dtype=ops.dtype(data)), ops.shape(data))
        ldj = -ops.log(ops.abs(std))
        if inverse:
            ldj = -ldj
        return self._sum_except_batch(ldj)
