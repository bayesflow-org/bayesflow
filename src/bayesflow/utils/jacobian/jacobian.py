from collections.abc import Callable

import keras
import numpy as np

from bayesflow.types import Tensor

from .vjp import vjp


def jacobian(f: Callable[[Tensor], Tensor], x: Tensor, return_output: bool = False):
    """
    Compute the Jacobian matrix of f with respect to x.

    Parameters
    ----------
    f : callable
        The function to be differentiated.
    x : Tensor of shape (..., D_in)
        The input tensor to f.
    return_output : bool, optional
        Whether to return the output of f(x) along with the Jacobian matrix.
        Default: False

    Returns
    -------
    Tensor of shape (..., D_out, D_in)
        The Jacobian matrix of f with respect to x.

    2-tuple of tensors
        1. The output of f(x) (if return_output is True)
        2. Tensor of shape (..., D_out, D_in)
            The Jacobian matrix of f with respect to x.

    """
    fx, vjp_fn = vjp(f, x, return_output=True)

    batch_shape = keras.ops.shape(x)[:-1]
    batch_size = np.prod(batch_shape)

    rows = keras.ops.shape(fx)[-1]
    cols = keras.ops.shape(x)[-1]

    jac = keras.ops.zeros((*batch_shape, rows, cols))

    for col in range(cols):
        projector = np.zeros(keras.ops.shape(x), dtype=keras.ops.dtype(x))
        projector[..., col] = 1.0
        projector = keras.ops.convert_to_tensor(projector)

        # jac[..., col] = vjp_fn(projector)
        indices = np.stack(list(np.ndindex(batch_shape + (rows,))))
        indices = np.concatenate([indices, np.full((batch_size * rows, 1), col)], axis=1)
        indices = keras.ops.convert_to_tensor(indices)

        updates = vjp_fn(projector)
        updates = keras.ops.reshape(updates, (-1,))
        jac = keras.ops.scatter_update(jac, indices, updates)

    if return_output:
        return fx, jac

    return jac
