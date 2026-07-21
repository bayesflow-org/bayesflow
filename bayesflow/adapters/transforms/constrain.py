import keras.ops as ops

from bayesflow.utils.serialization import serializable, serialize
from bayesflow.types import Tensor
from bayesflow.utils.keras_utils import (
    inverse_softplus,
)
from .elementwise_transform import ElementwiseTransform


@serializable("bayesflow.adapters")
class Constrain(ElementwiseTransform):
    """
    Constrains neural network predictions of a data variable to specified bounds.

    Parameters
    ----------
    * : str
        String containing the name of the data variable to be transformed e.g. "sigma". See examples below.

    lower : int or float or Tensor, optional
        Lower bound for named data variable.
    upper : int or float or Tensor, optional
        Upper bound for named data variable.
    method : str, optional
        Method by which to shrink the network predictions space to specified bounds. Choose from
        - Double bounded methods: sigmoid, expit, (default = sigmoid)
        - Lower bound only methods: softplus, exp, (default = softplus)
        - Upper bound only methods: softplus, exp, (default = softplus)
    inclusive : {'both', 'lower', 'upper', 'none'}, optional
        Indicates which bounds are inclusive (or exclusive).
        - "both" (default): Both lower and upper bounds are inclusive.
        - "lower": Lower bound is inclusive, upper bound is exclusive.
        - "upper": Lower bound is exclusive, upper bound is inclusive.
        - "none": Both lower and upper bounds are exclusive.
    epsilon : float, optional
        Small value to ensure inclusive bounds are not violated.
        Current default is 1e-15 as this ensures finite outcomes
        with the default transformations applied to data exactly at the boundaries.

    Examples
    --------

    1) Let sigma be the standard deviation of a normal distribution,
    then sigma should always be greater than zero.

    >>> adapter = (
        bf.Adapter()
        .constrain("sigma", lower=0)
    )

    2) Suppose p is the parameter for a binomial distribution where p must be in
    [0,1] then we would constrain the neural network to estimate p in the following way.

    >>> import bayesflow as bf
    >>> adapter = bf.Adapter()
    >>> adapter.constrain("p", lower=0, upper=1, method="sigmoid", inclusive="both")
    """

    def __init__(
        self,
        *,
        lower: int | float | Tensor = None,
        upper: int | float | Tensor = None,
        method: str = "default",
        inclusive: str = "both",
        epsilon: float = 1e-15,
    ):
        super().__init__()

        if lower is None and upper is None:
            raise ValueError("At least one of 'lower' or 'upper' must be provided.")

        if lower is not None and upper is not None:
            # double bounded case
            if ops.any(lower >= upper):
                raise ValueError("The lower bound must be strictly less than the upper bound.")

            match method:
                case "default" | "sigmoid" | "expit" | "logit":

                    def constrain(x):
                        return (upper - lower) * ops.sigmoid(x) + lower

                    def unconstrain(x):
                        z = (x - lower) / (upper - lower)
                        return ops.log(z) - ops.log1p(-z)

                    def ldj(x):
                        z = (x - lower) / (upper - lower)
                        return -ops.log(z) - ops.log1p(-z) - ops.cast(ops.log((upper - lower)), dtype=ops.dtype(z))

                case str() as name:
                    raise ValueError(f"Unsupported method name for double bounded constraint: '{name}'.")
                case other:
                    raise TypeError(f"Expected a method name, got {other!r}.")
        elif lower is not None:
            # lower bounded case
            match method:
                case "default" | "softplus":

                    def constrain(x):
                        return ops.softplus(x) + lower

                    def unconstrain(x):
                        return inverse_softplus(x - lower)

                    def ldj(x):
                        y = x - lower
                        return ops.subtract(y, ops.log(ops.expm1(y)))

                case "exp" | "log":

                    def constrain(x):
                        return ops.exp(x) + lower

                    def unconstrain(x):
                        return ops.log(x - lower)

                    def ldj(x):
                        return -ops.log(x - lower)

                case str() as name:
                    raise ValueError(f"Unsupported method name for single bounded constraint: '{name}'.")
                case other:
                    raise TypeError(f"Expected a method name, got {other!r}.")
        else:
            # upper bounded case
            match method:
                case "default" | "softplus":

                    def constrain(x):
                        return -ops.softplus(-x) + upper

                    def unconstrain(x):
                        return -inverse_softplus(-(x - upper))

                    def ldj(x):
                        y = upper - x
                        return ops.subtract(y, ops.log(ops.expm1(y)))

                case "exp" | "log":

                    def constrain(x):
                        return -ops.exp(-x) + upper

                    def unconstrain(x):
                        return -ops.log(upper - x)

                    def ldj(x):
                        return -ops.log(upper - x)

                case str() as name:
                    raise ValueError(f"Unsupported method name for single bounded constraint: '{name}'.")
                case other:
                    raise TypeError(f"Expected a method name, got {other!r}.")

        self.lower = lower
        self.upper = upper
        self.method = method
        self.inclusive = inclusive
        self.epsilon = epsilon

        self.constrain = constrain
        self.unconstrain = unconstrain
        self.ldj = ldj

        # do this last to avoid serialization issues
        match inclusive:
            case "lower":
                if lower is not None:
                    lower = lower - epsilon
            case "upper":
                if upper is not None:
                    upper = upper + epsilon
            case True | "both":
                if lower is not None:
                    lower = lower - epsilon
                if upper is not None:
                    upper = upper + epsilon
            case False | None | "none":
                pass
            case other:
                raise ValueError(f"Unsupported value for 'inclusive': {other!r}.")

    def get_config(self) -> dict:
        config = {
            "lower": self.lower,
            "upper": self.upper,
            "method": self.method,
            "inclusive": self.inclusive,
            "epsilon": self.epsilon,
        }
        return serialize(config)

    def forward(self, data: Tensor, **kwargs) -> Tensor:
        # forward means data space -> network space, so unconstrain the data
        return self.unconstrain(data)

    def inverse(self, data: Tensor, **kwargs) -> Tensor:
        # inverse means network space -> data space, so constrain the data
        return self.constrain(data)

    def log_det_jac(self, data: Tensor, inverse: bool = False, **kwargs) -> Tensor:
        ldj = self.ldj(data)
        if inverse:
            ldj = -ldj
        return self._sum_except_batch(ldj)
