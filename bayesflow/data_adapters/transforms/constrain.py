from keras.saving import (
    register_keras_serializable as serializable,
)
import numpy as np

from bayesflow.utils.numpy_utils import (
    inverse_sigmoid,
    inverse_softplus,
    sigmoid,
    softplus,
)

from .elementwise_transform import ElementwiseTransform


@serializable(package="bayesflow.data_adapters")
class Constrain(ElementwiseTransform):
    def __init__(
        self, *, lower: int | float | np.ndarray = None, upper: int | float | np.ndarray = None, method: str = "default"
    ):
        super().__init__()

        if lower is None and upper is None:
            raise ValueError("At least one of 'lower' or 'upper' must be provided.")

        if lower is not None and upper is not None:
            # double bounded case
            if np.any(lower >= upper):
                raise ValueError("The lower bound must be strictly less than the upper bound.")

            match method:
                case "default" | "sigmoid" | "expit" | "logit":

                    def constrain(x):
                        return (upper - lower) * sigmoid(x) + lower

                    def unconstrain(x):
                        return inverse_sigmoid((x - lower) / (upper - lower))
                case str() as name:
                    raise ValueError(f"Unsupported method name for double bounded constraint: '{name}'.")
                case other:
                    raise TypeError(f"Expected a method name, got {other!r}.")
        elif lower is not None:
            # lower bounded case
            match method:
                case "default" | "softplus":

                    def constrain(x):
                        return softplus(x) + lower

                    def unconstrain(x):
                        return inverse_softplus(x - lower)
                case "exp" | "log":

                    def constrain(x):
                        return np.exp(x) + lower

                    def unconstrain(x):
                        return np.log(x - lower)
                case str() as name:
                    raise ValueError(f"Unsupported method name for single bounded constraint: '{name}'.")
                case other:
                    raise TypeError(f"Expected a method name, got {other!r}.")
        else:
            # upper bounded case
            match method:
                case "default" | "softplus":

                    def constrain(x):
                        return -softplus(-x) + upper

                    def unconstrain(x):
                        return -inverse_softplus(-(x - upper))
                case "exp" | "log":

                    def constrain(x):
                        return -np.exp(-x) + upper

                    def unconstrain(x):
                        return -np.log(-x + upper)
                case str() as name:
                    raise ValueError(f"Unsupported method name for single bounded constraint: '{name}'.")
                case other:
                    raise TypeError(f"Expected a method name, got {other!r}.")

        self.lower = lower
        self.upper = upper

        self.method = method

        self.constrain = constrain
        self.unconstrain = unconstrain

    @classmethod
    def from_config(cls, config: dict, custom_objects=None) -> "Constrain":
        return cls(**config)

    def get_config(self) -> dict:
        return {
            "lower": self.lower,
            "upper": self.upper,
            "method": self.method,
        }

    def forward(self, data: np.ndarray, **kwargs) -> np.ndarray:
        # forward means data space -> network space, so unconstrain the data
        return self.unconstrain(data)

    def inverse(self, data: np.ndarray, **kwargs) -> np.ndarray:
        # inverse means network space -> data space, so constrain the data
        return self.constrain(data)
