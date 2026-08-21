import warnings
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
    f : Callable
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
    warnings.warn(
        "`jacobian` is deprecated; we are working on moving these utilities upstream or into their own module "
        "with improved signatures.",
        DeprecationWarning,
        stacklevel=2,
    )
    fx, vjp_fn = vjp(f, x, return_output=True)

    cols = keras.ops.shape(x)[-1]

    jac_columns = []
    for col in range(cols):
        projector = np.zeros(keras.ops.shape(x), dtype=keras.ops.dtype(x))
        projector[..., col] = 1.0
        projector = keras.ops.convert_to_tensor(projector)

        # jac[..., col] = vjp_fn(projector)
        jac_columns.append(vjp_fn(projector))

    jac = keras.ops.stack(jac_columns, axis=-1)

    if return_output:
        return fx, jac

    return jac
