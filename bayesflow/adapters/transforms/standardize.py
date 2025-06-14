from collections.abc import Sequence
import warnings

import numpy as np

from bayesflow.utils.serialization import serializable, serialize

from .elementwise_transform import ElementwiseTransform


@serializable("bayesflow.adapters")
class Standardize(ElementwiseTransform):
    """
    Transform that when applied standardizes data using typical z-score standardization
    i.e. for some unstandardized data x the standardized version z would be

    >>> z = (x - mean(x)) / std(x)

    Parameters
    ----------
    mean : int or float, optional
        Specify a mean if known but will be estimated from data when not provided
    std : int or float, optional
        Specify a standard devation if known but will be estimated from data when not provided
    axis : int, optional
        A specific axis along which standardization should take place. By default
        standardization happens individually for each dimension
    momentum : float in (0,1)
        The momentum during training

    Examples
    --------
    1) Standardize all variables using their individually estimated mean and stds.

    >>> adapter = (
            bf.adapters.Adapter()
                .standardize()
        )


    2) Standardize all with same known mean and std.

    >>> adapter = (
            bf.adapters.Adapter()
                .standardize(mean = 5, sd = 10)
        )


    3) Mix of fixed and estimated means/stds. Suppose we have priors for "beta" and "sigma" where we
    know the means and stds. However for all other variables, the means and stds are unknown.
    Then standardize should be used in several stages specifying which variables to include or exclude.

    >>> adapter = (
            bf.adapters.Adapter()
                # mean fixed, std estimated
                .standardize(include = "beta", mean = 1)
                # both mean and SD fixed
                .standardize(include = "sigma", mean = 0.6, sd = 3)
                # both means and stds estimated for all other variables
                .standardize(exclude = ["beta", "sigma"])
        )
    """

    def __init__(
        self,
        mean: int | float | np.ndarray = None,
        std: int | float | np.ndarray = None,
        axis: int | Sequence[int] = None,
        momentum: float | None = 0.99,
    ):
        super().__init__()

        if mean is None or std is None:
            warnings.warn(
                "Dynamic standardization is deprecated and will be removed in later versions."
                "Instead, use the standardize argument of the approximator / workflow instance or provide "
                "fixed mean and std arguments. You may incur some redundant computations if you keep this transform.",
                FutureWarning,
            )

        self.mean = mean
        self.std = std

        if isinstance(axis, Sequence):
            # numpy hates lists
            axis = tuple(axis)
        self.axis = axis
        self.momentum = momentum

    def get_config(self) -> dict:
        config = {
            "mean": self.mean,
            "std": self.std,
            "axis": self.axis,
            "momentum": self.momentum,
        }
        return serialize(config)

    def forward(self, data: np.ndarray, stage: str = "inference", **kwargs) -> np.ndarray:
        if self.axis is None:
            self.axis = tuple(range(data.ndim - 1))

        if self.mean is None:
            self.mean = np.mean(data, axis=self.axis, keepdims=True)
        else:
            if self.momentum is not None and stage == "training":
                self.mean = self.momentum * self.mean + (1.0 - self.momentum) * np.mean(
                    data, axis=self.axis, keepdims=True
                )

        if self.std is None:
            self.std = np.std(data, axis=self.axis, keepdims=True, ddof=1)
        else:
            if self.momentum is not None and stage == "training":
                self.std = self.momentum * self.std + (1.0 - self.momentum) * np.std(
                    data, axis=self.axis, keepdims=True, ddof=1
                )

        mean = np.broadcast_to(self.mean, data.shape)
        std = np.broadcast_to(self.std, data.shape)

        return (data - mean) / std

    def inverse(self, data: np.ndarray, **kwargs) -> np.ndarray:
        if self.mean is None or self.std is None:
            raise RuntimeError("Cannot call `inverse` before calling `forward` at least once.")

        mean = np.broadcast_to(self.mean, data.shape)
        std = np.broadcast_to(self.std, data.shape)

        return data * std + mean

    def log_det_jac(self, data, inverse: bool = False, **kwargs) -> np.ndarray:
        std = np.broadcast_to(self.std, data.shape)
        ldj = -np.log(np.abs(std))
        if inverse:
            ldj = -ldj
        return np.sum(ldj, axis=tuple(range(1, ldj.ndim)))
