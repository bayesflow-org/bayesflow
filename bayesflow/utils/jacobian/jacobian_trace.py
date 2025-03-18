from collections.abc import Callable

import keras

from bayesflow.types import Tensor
from .jacobian import jacobian
from .vjp import vjp


def jacobian_trace(
    f: Callable[[Tensor], Tensor],
    x: Tensor,
    max_steps: int = None,
    return_output: bool = False,
    seed: int | keras.random.SeedGenerator = None,
):
    """Compute or estimate the trace of the Jacobian matrix of f.

    Parameters
    ----------
    f : callable
        The function to be differentiated.
    x :  Tensor of shape (n, ..., d)
        The input tensor to f.
    max_steps : int, optional
        The maximum number of steps to use for the estimate.
        If this does not exceed the dimensionality of f(x), use Hutchinson's algorithm to
        return an unbiased estimate of the Jacobian trace.
        Otherwise, perform an exact computation.
        Default: None
    return_output : bool, optional
        Whether to return the output of f(x) along with the trace of the Jacobian.
        Default: False
    seed : int or keras SeedGenerator, optional
        The seed to use for hutchinson trace estimation. Only has an effect when max_steps < d.

    Returns
    -------
    2-tuple of tensors:
        1. The output of f(x) (if return_output is True)
        2. Tensor of shape (n,)
            An unbiased estimate or the exact trace of the Jacobian of f.
    """
    dims = keras.ops.shape(x)[-1]

    if max_steps is None or dims <= max_steps:
        fx, jac = jacobian(f, x, return_output=True)
        trace = keras.ops.trace(jac, axis1=-2, axis2=-1)
    else:
        fx, trace = _hutchinson(f, x, steps=max_steps, return_output=True, seed=seed)

    if return_output:
        return fx, trace

    return trace


def _hutchinson(
    f: callable, x: Tensor, steps: int = 1, return_output: bool = False, seed: int | keras.random.SeedGenerator = None
):
    """Estimate the trace of the Jacobian matrix of f using Hutchinson's algorithm.

    :param f: The function to be differentiated.

    :param x: Tensor of shape (n,..., d)
        The input tensor to f.

    :param steps: The number of steps to use for the estimate.
        Higher values yield better precision.
        Default: 1

    :return: 2-tuple of tensors:
        1. The output of f(x)
        2. Tensor of shape (n,)
            An unbiased estimate of the trace of the Jacobian matrix of f.
    """
    shape = keras.ops.shape(x)
    trace = keras.ops.zeros(shape[:-1])

    fx, vjp_fn = vjp(f, x, return_output=True)

    for _ in range(steps):
        projector = keras.random.normal(shape, seed=seed)
        trace += keras.ops.sum(vjp_fn(projector) * projector, axis=-1)

    if return_output:
        return fx, trace

    return trace
