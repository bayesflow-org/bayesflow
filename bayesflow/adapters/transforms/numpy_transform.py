import numpy as np

from bayesflow.utils.serialization import serializable, serialize

from .elementwise_transform import ElementwiseTransform


@serializable("bayesflow.adapters")
class NumpyTransform(ElementwiseTransform):
    """
    A class to apply element-wise transformations using plain NumPy functions.

    Attributes
    ----------
    _forward : str
        The name of the NumPy function to apply in the forward transformation.
    _inverse : str
        The name of the NumPy function to apply in the inverse transformation.
    """

    #: Dict of `np.ufunc` that support automatic selection of their inverse.
    INVERSE_METHODS = {
        np.arctan: np.tan,
        np.exp: np.log,
        np.expm1: np.log1p,
        np.square: np.sqrt,
        np.reciprocal: np.reciprocal,
    }
    # ensure the map is symmetric
    INVERSE_METHODS |= {v: k for k, v in INVERSE_METHODS.items()}

    def __init__(self, forward: str, inverse: str = None):
        """
        Initializes the NumpyTransform with specified forward and inverse functions.

        Parameters
        ----------
        forward : str
            The name of the NumPy function to use for the forward transformation.
        inverse : str, optional
            The name of the NumPy function to use for the inverse transformation.
            By default, the inverse is inferred from the forward argument for supported methods.
        """
        super().__init__()

        if isinstance(forward, str):
            forward = getattr(np, forward)

        if not isinstance(forward, np.ufunc):
            raise ValueError("Forward transformation must be a NumPy Universal Function (ufunc).")

        if inverse is None:
            if forward not in self.INVERSE_METHODS:
                raise ValueError(f"Cannot infer inverse for method {forward!r}")

            inverse = self.INVERSE_METHODS[forward]

        if isinstance(inverse, str):
            inverse = getattr(np, inverse)

        if not isinstance(inverse, np.ufunc):
            raise ValueError("Inverse transformation must be a NumPy Universal Function (ufunc).")

        self._forward = forward
        self._inverse = inverse

    def get_config(self) -> dict:
        return serialize({"forward": self._forward.__name__, "inverse": self._inverse.__name__})

    def forward(self, data: dict[str, any], **kwargs) -> dict[str, any]:
        return self._forward(data)

    def inverse(self, data: np.ndarray, **kwargs) -> np.ndarray:
        return self._inverse(data)

    def log_det_jac(self, data, inverse=False, **kwargs):
        raise NotImplementedError("log determinant of the Jacobian of the numpy transforms are not implemented yet")
