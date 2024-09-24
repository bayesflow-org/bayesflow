from collections.abc import Sequence
from keras.saving import (
    deserialize_keras_object as deserialize,
    register_keras_serializable as serializable,
    serialize_keras_object as serialize,
)
import numpy as np

from bayesflow.utils.numpy_utils import (
    inverse_sigmoid,
    inverse_softplus,
    sigmoid,
    softplus,
)

from .lambda_transform import LambdaTransform


@serializable(package="bayesflow.data_adapters")
class ConstrainBounded(LambdaTransform):
    """Constrains a parameter with a lower and/or upper bound."""

    def __init__(
        self,
        parameters: str | Sequence[str] | None = None,
        /,
        *,
        lower: np.ndarray = None,
        upper: np.ndarray = None,
        method: str,
    ):
        self.lower = lower
        self.upper = upper
        self.method = method

        if lower is None and upper is None:
            raise ValueError("At least one of 'lower' or 'upper' must be provided.")

        if lower is not None and upper is not None:
            if np.any(lower >= upper):
                raise ValueError("The lower bound must be strictly less than the upper bound.")

            # double bounded case
            match method:
                case "clip":

                    def constrain(x):
                        return np.clip(x, lower, upper)

                    def unconstrain(x):
                        # not bijective
                        return x
                case "sigmoid":

                    def constrain(x):
                        return (upper - lower) * sigmoid(x) + lower

                    def unconstrain(x):
                        return inverse_sigmoid((x - lower) / (upper - lower))
                case str() as name:
                    raise ValueError(f"Unsupported method name for double bounded constraint: '{name}'.")
                case other:
                    raise TypeError(f"Expected a method name, got {other!r}.")
        else:
            # single bounded case
            if lower is not None:
                match method:
                    case "clip":

                        def constrain(x):
                            return np.clip(x, lower, np.inf)

                        def unconstrain(x):
                            # not bijective
                            return x
                    case "softplus":

                        def constrain(x):
                            return softplus(x) + lower

                        def unconstrain(x):
                            return inverse_softplus(x - lower)
                    case "exp":

                        def constrain(x):
                            return np.exp(x) + lower

                        def unconstrain(x):
                            return np.log(x - lower)
                    case str() as name:
                        raise ValueError(f"Unsupported method name for single bounded constraint: '{name}'.")
                    case other:
                        raise TypeError(f"Expected a method name, got {other!r}.")
            else:
                match method:
                    case "clip":

                        def constrain(x):
                            return np.clip(x, -np.inf, upper)

                        def unconstrain(x):
                            # not bijective
                            return x
                    case "softplus":

                        def constrain(x):
                            return -softplus(-x) + upper

                        def unconstrain(x):
                            return -inverse_softplus(-(x - upper))
                    case "exp":

                        def constrain(x):
                            return -np.exp(-x) + upper

                        def unconstrain(x):
                            return -np.log(-x + upper)
                    case str() as name:
                        raise ValueError(f"Unsupported method name for single bounded constraint: '{name}'.")
                    case other:
                        raise TypeError(f"Expected a method name, got {other!r}.")

        super().__init__(parameters, forward=unconstrain, inverse=constrain)

    @classmethod
    def from_config(cls, config: dict, custom_objects=None) -> "ConstrainBounded":
        return cls(
            deserialize(config["parameters"], custom_objects),
            lower=deserialize(config["lower"], custom_objects),
            upper=deserialize(config["upper"], custom_objects),
            method=deserialize(config["method"], custom_objects),
        )

    def get_config(self) -> dict:
        return {
            "parameters": serialize(self.parameters),
            "lower": serialize(self.lower),
            "upper": serialize(self.upper),
            "method": serialize(self.method),
        }
