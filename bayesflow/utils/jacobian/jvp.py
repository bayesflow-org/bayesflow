import warnings
from collections.abc import Callable

import keras

from bayesflow._backend import jvp as _backend_jvp
from bayesflow.types import Tensor


def jvp(
    f: Callable, x: Tensor | tuple[Tensor, ...], tangents: Tensor | tuple[Tensor, ...], return_output: bool = False
):
    """Compute the Jacobian-vector product of f at x with tangents."""
    warnings.warn(
        "jvp is deprecated; we are working on moving these utilities upstream or into their own module with "
        "improved signatures.",
        DeprecationWarning,
        stacklevel=2,
    )
    if keras.ops.is_tensor(x):
        x = (x,)

    if keras.ops.is_tensor(tangents):
        tangents = (tangents,)

    fx, jvp_out = _backend_jvp(f, list(x), list(tangents))

    if return_output:
        return fx, jvp_out

    return jvp_out
