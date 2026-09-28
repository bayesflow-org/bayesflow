import keras
import keras.ops as ops
import numpy as np
import math
from bayesflow.utils.serialization import serializable, serialize
from bayesflow.types import Tensor

from .elementwise_transform import ElementwiseTransform


@serializable("bayesflow.adapters")
class CorrelationMatrix(ElementwiseTransform):
    """
    Constrains neural network predictions of a variable to a valid (symmetric, positive definite,
    unit diagonal) correlation matrix, using the transforms explained by [1].

    The unconstrained representation is a flat vector `y` of K * (K - 1) / 2 entries.

    Parameters
    ----------
    cholesky : bool, optional
        Whether the *constrained* side of the transform is the lower Cholesky factor `x`,
        rather than the full correlation matrix `x @ x.T`.
        Default is False (use the full correlation matrix).

    References
    ----------
    [1] Lewandowski, D., Kurowicka, D., & Joe, H. (2009).
        Generating random correlation matrices based on vines and extended onion method.
        Journal of Multivariate Analysis, 100(9), 1989-2001. https://doi.org/10.1016/j.jmva.2009.04.008

    Examples
    --------
    >>> adapter = bf.Adapter().correlation_matrix("corr_x")
    """

    def __init__(self, *, cholesky: bool = False):
        super().__init__()
        self.cholesky = cholesky

    def get_config(self) -> dict:
        return serialize({"cholesky": self.cholesky})

    def forward(self, data: Tensor, **kwargs) -> Tensor:
        if not self.cholesky:
            data = ops.cholesky(data)

        z = self._mat2vec(data) / self._normalization(data)
        epsilon = keras.config.epsilon()
        z = ops.clip(z, -1.0 + epsilon, 1.0 - epsilon)
        return ops.arctanh(z)

    def inverse(self, data: Tensor, **kwargs) -> Tensor:
        P = ops.shape(data)[-1]
        K = (math.isqrt(1 + 8 * P) + 1) // 2

        z = self._vec2mat(ops.tanh(data), K)
        identity = ops.eye(K, dtype=z.dtype)
        epsilon = keras.config.epsilon()

        # Build x column by column since column j depends on all previous columns j' < j
        columns = []
        # Keep track of the sum of squares of the previous columns
        sum_of_squares = ops.zeros_like(z[..., :, 0])
        for j in range(K):
            normalization = ops.sqrt(ops.maximum(1.0 - sum_of_squares, epsilon))

            # if i > j: x[i,j] = z[i,j] * normalization
            # if i = j: x[i,j] = normalization
            # if i < j: x[i,j] = 0
            x_j = (z[..., :, j] + identity[:, j]) * normalization

            columns.append(x_j)
            sum_of_squares = sum_of_squares + ops.square(x_j)

        x = ops.stack(columns, axis=-1)

        if not self.cholesky:
            x = ops.matmul(x, ops.swapaxes(x, -1, -2))
            # set diagonals to 1 exactly
            x = x * (1.0 - identity) + identity
            return x
        return x

    def log_det_jac(self, data: Tensor, inverse: bool = False, **kwargs) -> Tensor:
        K = ops.shape(data)[-1]
        if not self.cholesky:
            data = ops.cholesky(data)

        normalization = self._normalization(data)
        z = self._mat2vec(data) / normalization
        epsilon = keras.config.epsilon()
        z = ops.clip(z, -1.0 + epsilon, 1.0 - epsilon)

        ldj = ops.sum(ops.log1p(-ops.square(z)), axis=-1)
        ldj = ldj + ops.sum(ops.log(normalization), axis=-1)

        if not self.cholesky:
            diag = ops.diagonal(data, axis1=-2, axis2=-1)
            weight = ops.cast(K - 1, diag.dtype) - ops.arange(K, dtype=diag.dtype)
            ldj = ldj + ops.sum(weight * ops.log(diag), axis=-1)

        if not inverse:
            ldj = -ldj

        return self._sum_except_batch(ldj)

    def _normalization(self, x: Tensor) -> Tensor:
        """Computes sqrt(1 - sum_{j' < j} x[i, j']^2) for all i > j, returns a flat vector filled by row."""
        x2 = ops.square(x)
        # cumsum over columns includes x[i, j] itself, so subtract it to sum over j' < j only
        normalization = 1.0 - (ops.cumsum(x2, axis=-1) - x2)
        return ops.sqrt(self._mat2vec(normalization))

    @staticmethod
    def _mat2vec(z: Tensor) -> Tensor:
        """Extracts the strictly lower triangle of `z`, filled by row, into a flat vector."""
        K = ops.shape(z)[-1]
        batch_shape = tuple(ops.shape(z)[:-2])
        rows, cols = np.tril_indices(K, k=-1)
        return ops.take(ops.reshape(z, batch_shape + (K * K,)), K * rows + cols, axis=-1)

    @staticmethod
    def _vec2mat(z: Tensor, K: int) -> Tensor:
        """Places a flat vector into the strictly lower triangle of a K x K matrix, filled by row."""
        batch_shape = tuple(ops.shape(z)[:-1])
        rows, cols = np.tril_indices(K, k=-1)

        idx_map = np.zeros(K * K, dtype="int32")
        idx_map[rows * K + cols] = 1 + np.arange(len(rows))
        padded = ops.concatenate([ops.zeros_like(z[..., :1]), z], axis=-1)
        return ops.reshape(ops.take(padded, idx_map, axis=-1), batch_shape + (K, K))
