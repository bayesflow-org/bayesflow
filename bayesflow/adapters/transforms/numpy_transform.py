import numpy as np

from bayesflow.utils import filter_kwargs
from .elementwise_transform import ElementwiseTransform


class NumpyTransform(ElementwiseTransform):
    """
    A class to apply element-wise transformations using plain NumPy functions.

    Attributes:
    ----------
    _forward : str
        The name of the NumPy function to apply in the forward transformation.
    _inverse : str
        The name of the NumPy function to apply in the inverse transformation.
    """

    INVERSE_METHODS = {
        "arctan": "tan",
        "exp": "log",
        "expm1": "log1p",
        "square": "sqrt",
        "reciprocal": "reciprocal",
    }
    # ensure the map is symmetric
    INVERSE_METHODS |= {v: k for k, v in INVERSE_METHODS.items()}

    def __init__(self, forward: np.ufunc | str, inverse: np.ufunc | str = None):
        """
        Initializes the NumpyTransform with specified forward and inverse functions.

        Parameters:
        ----------
        forward : str
            The name of the NumPy function to use for the forward transformation.
        inverse : str
            The name of the NumPy function to use for the inverse transformation.
            By default, the inverse is inferred from the forward argument for supported methods.
        """
        super().__init__()

        if isinstance(forward, np.ufunc):
            forward = forward.__name__

        if inverse is None:
            if forward not in self.INVERSE_METHODS:
                raise ValueError(f"Cannot infer inverse for method {forward!r}")

            inverse = self.INVERSE_METHODS[forward]
        elif isinstance(inverse, np.ufunc):
            inverse = inverse.__name__

        if forward not in dir(np):
            raise ValueError(f"Method {forward!r} not found in numpy version {np.__version__}")

        if inverse not in dir(np):
            raise ValueError(f"Method {inverse!r} not found in numpy version {np.__version__}")

        self._forward = forward
        self._inverse = inverse

    @classmethod
    def from_config(cls, config: dict, custom_objects=None) -> "ElementwiseTransform":
        return cls(
            forward=config["forward"],
            inverse=config["inverse"],
        )

    def get_config(self) -> dict:
        return {"forward": self._forward, "inverse": self._inverse}

    def forward(self, data: dict[str, any], **kwargs) -> dict[str, any]:
        forward = getattr(np, self._forward)
        kwargs = filter_kwargs(kwargs, forward)
        return forward(data, **kwargs)

    def inverse(self, data: np.ndarray, **kwargs) -> np.ndarray:
        inverse = getattr(np, self._inverse)
        kwargs = filter_kwargs(kwargs, inverse)
        return inverse(data, **kwargs)
