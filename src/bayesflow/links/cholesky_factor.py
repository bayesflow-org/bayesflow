import keras

from bayesflow.types import Tensor
from bayesflow.utils import layer_kwargs, fill_triangular_matrix, positive_diag
from bayesflow.utils.serialization import serializable


@serializable("bayesflow.links")
class CholeskyFactor(keras.Layer):
    """Activation function to link from a flat tensor to a lower triangular matrix with positive diagonal."""

    def __init__(self, **kwargs):
        super().__init__(**layer_kwargs(kwargs))

    def call(self, inputs: Tensor) -> Tensor:
        # form a cholesky factor
        L = fill_triangular_matrix(inputs)
        L = positive_diag(L)

        return L

    def compute_output_shape(self, input_shape):
        m = input_shape[-1]
        n = int((0.25 + 2.0 * m) ** 0.5 - 0.5)
        return input_shape[:-1] + (n, n)

    def compute_input_shape(self, output_shape):
        """
        Returns the shape of parameterization of a Cholesky factor triangular matrix.

        There are :math:`m` nonzero elements of a lower triangular :math:`n \\times n` matrix with
        :math:`m = n (n + 1) / 2`, so for output shape (..., n, n) the returned shape is (..., m).

        Examples
        --------
        >>> PositiveDefinite().compute_input_shape((None, 3, 3))
        6
        """
        n = output_shape[-1]
        m = int(n * (n + 1) / 2)
        return output_shape[:-2] + (m,)
