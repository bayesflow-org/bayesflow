from collections.abc import Sequence
import numpy as np


class Transform:
    """Implements typical data transformations that can be applied as part of the adapter pipeline."""

    def __call__(self, data: dict[str, np.ndarray], inverse: bool = False) -> dict[str, np.ndarray]:
        raise NotImplementedError


class ElementwiseTransform(Transform):
    """Intermediate layer for transforms that are applied on a per-parameter basis."""

    def __init__(self, parameters: str | Sequence[str] | None = None, /):
        self.parameters = parameters

    def __call__(self, data: dict[str, np.ndarray], inverse: bool = False) -> dict[str, np.ndarray]:
        """Apply the transform to the data"""
        data = data.copy()

        if self.parameters is None:
            # apply to all parameters
            parameters = list(data.keys())
        elif isinstance(self.parameters, str):
            # apply just to this parameter
            parameters = [self.parameters]
        else:
            # apply to all given parameters
            parameters = self.parameters

        for parameter in parameters:
            if data.get(parameter) is None:
                # skip when in partial configuration
                continue

            if inverse:
                data[parameter] = self.inverse(parameter, data[parameter])
            else:
                data[parameter] = self.forward(parameter, data[parameter])

        return data

    def forward(self, parameter_name: str, parameter_value: np.ndarray) -> np.ndarray:
        """Implements the forward direction of the transform, i.e., user/data space -> network/latent space"""
        raise NotImplementedError

    def inverse(self, parameter_name: str, parameter_value: np.ndarray) -> np.ndarray:
        """Implements the inverse direction of the transform, i.e., network/latent space -> userdata space"""
        raise NotImplementedError
