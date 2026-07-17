import keras.ops as ops

from bayesflow.types import Tensor
from bayesflow.utils.serialization import serializable, serialize

from .elementwise_transform import ElementwiseTransform


@serializable("bayesflow.adapters")
class KerasTransform(ElementwiseTransform):
    """
    A class to apply element-wise transformations using named `keras.ops` functions.

    Attributes
    ----------
    _forward : str
        The name of the `keras.ops` function to apply in the forward transformation.
    _inverse : str
        The name of the `keras.ops` function to apply in the inverse transformation.
    """

    #: Dict of function names that support automatic selection of their inverse.
    INVERSE_METHODS = {
        "arctan": "tan",
        "exp": "log",
        "expm1": "log1p",
        "square": "sqrt",
        "reciprocal": "reciprocal",
    }
    # ensure the map is symmetric
    INVERSE_METHODS |= {v: k for k, v in INVERSE_METHODS.items()}

    def __init__(self, forward: str, inverse: str = None):
        """
        Initializes the KerasTransform with specified forward and inverse functions.

        Parameters
        ----------
        forward : str
            The name of the `keras.ops` function to use for the forward transformation.
        inverse : str, optional
            The name of the `keras.ops` function to use for the inverse transformation.
            By default, the inverse is inferred from the forward argument for supported methods.
        """
        super().__init__()

        self._forward_name = self._validate_name(forward)

        if inverse is None:
            if self._forward_name not in self.INVERSE_METHODS:
                raise ValueError(f"Cannot infer inverse for method {forward!r}")

            inverse = self.INVERSE_METHODS[self._forward_name]

        self._inverse_name = self._validate_name(inverse)

    @staticmethod
    def _validate_name(name: str) -> str:
        if not isinstance(name, str):
            raise ValueError("Transformation must be specified as the name of a keras.ops function.")

        if not callable(getattr(ops, name, None)):
            raise ValueError(f"keras.ops has no function named {name!r}.")

        return name

    def get_config(self) -> dict:
        return serialize({"forward": self._forward_name, "inverse": self._inverse_name})

    def forward(self, data: Tensor, **kwargs) -> Tensor:
        return getattr(ops, self._forward_name)(data)

    def inverse(self, data: Tensor, **kwargs) -> Tensor:
        return getattr(ops, self._inverse_name)(data)

    def log_det_jac(self, data, inverse=False, **kwargs):
        raise NotImplementedError("log determinant of the Jacobian of the keras transforms are not implemented yet")
