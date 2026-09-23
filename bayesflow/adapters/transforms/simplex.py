import keras.ops as ops

from bayesflow.utils.serialization import serializable, serialize
from bayesflow.types import Tensor
from .elementwise_transform import ElementwiseTransform


@serializable("bayesflow.adapters")
class Simplex(ElementwiseTransform):
    """
    Constrains neural network predictions of a data variable to a unit simplex, so that the
    constrained representation is non-negative and sums to one along the given axis.

    Parameters
    ----------
    axis : int, optional
        The axis of the *simplex-constrained* data along which values sum to one over all K elements.
        The corresponding unconstrained representation has a size of K-1 along this axis.
    method : str, optional
        Method by which to transform between the K dimensional simplex space
        and the K-1 dimensional unconstrained space.
        - "default" / "simplex": sum-to-zero (orthogonal basis) transform followed by a softmax transform.
        Plain softmax is not invertible (shifting all values by a constant results in the same inverse).
        The sum-to-zero removes this issue.
        - "stick": stick-breaking logistic transform.
    """

    def __init__(self, *, axis: int = -1, method: str = "default"):
        self.axis = axis
        self.method = method

        match method:
            case "default" | "simplex":

                def sum_to_zero_basis(k, dtype):
                    # K x (K-1) matrix (Helmert contrast matrix)
                    j = ops.reshape(ops.arange(1, k, dtype=dtype), (1, k - 1))
                    i = ops.reshape(ops.arange(1, k + 1, dtype=dtype), (k, 1))
                    denom = ops.sqrt(j * (j + 1.0))
                    basis = ops.where(i <= j, 1.0 / denom, 0.0)
                    basis = ops.where(i == j + 1.0, -j / denom, basis)
                    return basis

                def sum_to_zero(x, inverse=False):
                    # forward: a zero sum K vector into a K-1 vector
                    # inverse: K-1 vector as a zero-sum K vector
                    if inverse:
                        K = ops.shape(x)[-1] + 1
                        basis = sum_to_zero_basis(K, ops.dtype(x))
                        return ops.matmul(x, ops.transpose(basis))
                    else:
                        K = ops.shape(x)[-1]
                        basis = sum_to_zero_basis(K, ops.dtype(x))
                        return ops.matmul(x, basis)

                def constrain(y):
                    y = ops.moveaxis(y, self.axis, -1)
                    zero_sum = sum_to_zero(y, inverse=True)
                    x = ops.softmax(zero_sum, axis=-1)
                    return ops.moveaxis(x, -1, self.axis)

                def unconstrain(x):
                    x = ops.moveaxis(x, self.axis, -1)
                    log_x = ops.log(x)
                    demeaned_log_x = log_x - ops.mean(log_x, axis=-1, keepdims=True)
                    y = sum_to_zero(demeaned_log_x)
                    return ops.moveaxis(y, -1, self.axis)

                def ldj(x):
                    return -ops.sum(ops.log(x), axis=self.axis)

            case "stick":

                def stick(x):
                    K = ops.shape(x)[-1]
                    r = ops.cumsum(x, axis=-1) - x
                    r = 1.0 - r[..., : K - 1]
                    z = x[..., : K - 1] / r
                    offset = ops.log(ops.arange(K - 1, 0, -1, dtype=ops.dtype(x)))
                    return z, r, offset

                def constrain(y):
                    y = ops.moveaxis(y, self.axis, -1)
                    K = ops.shape(y)[-1] + 1
                    offset = ops.log(ops.arange(K - 1, 0, -1, dtype=ops.dtype(y)))
                    z = ops.sigmoid(y - offset)
                    log_one_minus_z = ops.log1p(-z)
                    log_r = ops.cumsum(log_one_minus_z, axis=-1) - log_one_minus_z
                    x_first = z * ops.exp(log_r)
                    x_last = ops.exp(ops.sum(log_one_minus_z, axis=-1, keepdims=True))
                    x = ops.concatenate([x_first, x_last], axis=-1)
                    return ops.moveaxis(x, -1, self.axis)

                def unconstrain(x):
                    x = ops.moveaxis(x, self.axis, -1)
                    z, _, offset = stick(x)
                    y = ops.log(z) - ops.log1p(-z) + offset
                    return ops.moveaxis(y, -1, self.axis)

                def ldj(x):
                    x = ops.moveaxis(x, self.axis, -1)
                    z, r, _ = stick(x)
                    terms = -(ops.log(z) + ops.log1p(-z) + ops.log(r))
                    return ops.sum(terms, axis=-1)

            case str() as name:
                raise ValueError(f"Unsupported method name for simplex transform: '{name}'.")
            case other:
                raise TypeError(f"Expected a method name, got {other!r}.")

        self.constrain = constrain
        self.unconstrain = unconstrain
        self.ldj = ldj

    def get_config(self) -> dict:
        config = {
            "axis": self.axis,
            "method": self.method,
        }
        return serialize(config)

    def forward(self, data: Tensor, **kwargs) -> Tensor:
        return self.unconstrain(data)

    def inverse(self, data: Tensor, **kwargs) -> Tensor:
        return self.constrain(data)

    def log_det_jac(self, data: Tensor, inverse: bool = False, **kwargs) -> Tensor:
        ldj = self.ldj(data)
        if inverse:
            ldj = -ldj
        return self._sum_except_batch(ldj)
