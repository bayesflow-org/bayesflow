from collections.abc import Callable

from bayesflow._backend import vjp as _backend_vjp
from bayesflow.types import Tensor


def vjp(f: Callable[[Tensor], Tensor], x: Tensor, return_output: bool = False):
    """Compute the vector-Jacobian product of f at x."""
    fx, _vjp_fn = _backend_vjp(f, x)

    def vjp_fn(projector):
        # _backend vjp_fn returns a tuple (one gradient per primal).
        # Unwrap the single-primal case to preserve the original scalar API.
        return _vjp_fn(projector)[0]

    if return_output:
        return fx, vjp_fn

    return vjp_fn
