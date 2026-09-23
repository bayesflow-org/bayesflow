from bayesflow.adapters.transforms import Constrain
import keras.ops as ops
import numpy as np
import math
from bayesflow.utils.serialization import serializable, serialize
from bayesflow.types import Tensor

from .elementwise_transform import ElementwiseTransform


@serializable("bayesflow.adapters")
class CovarianceMatrix(ElementwiseTransform):
    """
    Constrains neural network predictions of a variable to a valid (symmetric,
    positive definite) covariance (or precision) matrix.

    The unconstrained representation is a flat vector holding the entries of the lower
    Cholesky factor `L` of the covariance matrix `Sigma = L @ L.T`: first the K diagonal
    entries of `L` (passed through a lower-bounded :py:class:`~transforms.Constrain`
    transform to keep them positive), followed by the K * (K - 1) / 2 entries from the lower triangular.

    Parameters
    ----------
    cholesky : bool, optional
        Whether the *constrained* side of the transform is the Cholesky factor `L`,
        rather than the full covariance matrix `Sigma = L @ L.T`.
        Default is False (use the full covariance matrix).
    diag_kwargs : dict, optional
        Keyword arguments forwarded to the :py:class:`~transforms.Constrain` transform
        that constrains the diagonal of `L` to be positive.
        The `lower` bound is always fixed to 0.0.

    Examples
    --------
    >>> adapter = bf.Adapter().covariance_matrix("cov_x", diag_kwargs={"method": "exp"})
    """

    def __init__(self, *, cholesky: bool = False, diag_kwargs: dict = None):
        super().__init__()
        self.cholesky = cholesky
        self.diag_kwargs = diag_kwargs or {}
        self.diag_kwargs["lower"] = 0.0
        self.diag_transform = Constrain(**self.diag_kwargs)

    def forward(self, data: Tensor, **kwargs) -> Tensor:
        K = ops.shape(data)[-1]
        if not self.cholesky:
            data = ops.cholesky(data)

        diag = ops.diagonal(data, axis1=-2, axis2=-1)
        diag = self.diag_transform(diag)

        # extract flattened off-diagonal
        rows, cols = np.tril_indices(K, k=-1)
        idx = K * rows + cols
        data_flat = ops.reshape(data, (-1, K * K))
        lower = ops.take(data_flat, idx, axis=-1)

        return ops.concatenate([diag, lower], axis=-1)

    def inverse(self, data: Tensor, **kwargs) -> Tensor:
        P = ops.shape(data)[-1]
        K = (math.isqrt(1 + 8 * P) - 1) // 2

        diag = self.diag_transform(data[..., :K], inverse=True)
        lower = data[..., K:]

        # Index map into [0, diag..., lower...] for each flattened (i, j) position
        rows, cols = np.tril_indices(K, k=-1)
        idx_map = np.zeros(K * K, dtype="int32")  # upper triangle -> slot 0 (zero)
        idx_map[np.arange(K) * (K + 1)] = 1 + np.arange(K)  # diagonal
        idx_map[rows * K + cols] = 1 + K + np.arange(len(rows))  # strictly lower

        padded = ops.concatenate([ops.zeros_like(diag[..., :1]), diag, lower], axis=-1)
        L = ops.reshape(ops.take(padded, idx_map, axis=-1), (-1, K, K))

        if not self.cholesky:
            return ops.matmul(L, ops.swapaxes(L, -1, -2))  # L @ L^T
        return L

    def get_config(self) -> dict:
        config = {
            "cholesky": self.cholesky,
            "diag_kwargs": self.diag_kwargs,
        }
        return serialize(config)

    def log_det_jac(self, data: Tensor, inverse: bool = False, **kwargs) -> Tensor:
        K = ops.shape(data)[-1]

        if not self.cholesky:
            data = ops.cholesky(data)

        diag = ops.diagonal(data, axis1=-2, axis2=-1)
        ldj = self.diag_transform.log_det_jac(diag, inverse=False)

        if not self.cholesky:
            # additional term for the cholesky transform
            weight = ops.cast(K, diag.dtype) - ops.arange(K, dtype=diag.dtype)
            chol_to_matrix_ldj = K * ops.log(ops.cast(2.0, diag.dtype)) + ops.sum(weight * ops.log(diag), axis=-1)
            ldj = ldj - chol_to_matrix_ldj

        if inverse:
            ldj = -ldj

        return ldj
