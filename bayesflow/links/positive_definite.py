import keras

from bayesflow.types import Tensor
from bayesflow.utils import layer_kwargs, fill_triangular_matrix
from bayesflow.utils.serialization import serializable

from warnings import warn


@serializable("bayesflow.links")
class PositiveDefinite(keras.Layer):
    """Activation function to link from flat elements of a lower triangular matrix to a positive definite matrix."""

    def __init__(self, **kwargs):
        super().__init__(**layer_kwargs(kwargs))
        self.built = True

        warn(
            "This class is deprecated. It was replaced by bayesflow.links.CholeskyFactor.",
            DeprecationWarning,
            stacklevel=2,
        )

    def call(self, inputs: Tensor) -> Tensor:
        # Build cholesky factor from inputs
        L = fill_triangular_matrix(inputs, positive_diag=True)

        # calculate positive definite matrix from cholesky factors
        psd = keras.ops.matmul(
            L,
            keras.ops.moveaxis(L, -2, -1),  # L transposed
        )
        return psd

    def compute_output_shape(self, input_shape):
        m = input_shape[-1]
        n = int((0.25 + 2.0 * m) ** 0.5 - 0.5)
        return input_shape[:-1] + (n, n)

    def compute_input_shape(self, output_shape):
        """
        Returns the shape of parameterization of a cholesky factor triangular matrix.

        There are m nonzero elements of a lower triangular nxn matrix with m = n * (n + 1) / 2.

        Example
        -------
        >>> PositiveDefinite().compute_output_shape((None, 3, 3))
        6
        """
        n = output_shape[-1]
        m = int(n * (n + 1) / 2)
        return output_shape[:-2] + (m,)
