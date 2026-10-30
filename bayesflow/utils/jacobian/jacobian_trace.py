import warnings
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
    projectors: Tensor = None,
):
    """Compute or estimate the trace of the Jacobian matrix of f.

    Parameters
    ----------
    f : Callable
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
        The seed to use for Hutchinson trace estimation. Only has an effect when max_steps < d.
    projectors : Tensor of shape (steps, n, ..., d), optional
        Fixed probe vectors for Hutchinson's algorithm. Pass the same projectors to every function
        evaluation of an ODE solve to keep the augmented dynamics deterministic.

    Returns
    -------
    2-tuple of tensors:
        1. The output of f(x) (if return_output is True)
        2. Tensor of shape (n,)
            An unbiased estimate or the exact trace of the Jacobian of f.
    """
    warnings.warn(
        "jacobian_trace is deprecated; we are working on moving these utilities upstream or into their own "
        "module with improved signatures.",
        DeprecationWarning,
        stacklevel=2,
    )

    dims = keras.ops.shape(x)[-1]

    if projectors is None and (max_steps is None or dims <= max_steps):
        fx, jac = jacobian(f, x, return_output=True)
        trace = keras.ops.trace(jac, axis1=-2, axis2=-1)
    else:
        fx, trace = _hutchinson(f, x, steps=max_steps, return_output=True, projectors=projectors, seed=seed)

    if return_output:
        return fx, trace

    return trace


def hutchinson_projectors(x: Tensor, steps: int | None, seed: int | keras.random.SeedGenerator = None) -> Tensor | None:
    """Sample Rademacher probe vectors of shape (steps, *x.shape) for Hutchinson's algorithm.

    Returns None if steps is None or not smaller than the dimensionality of x, in which case
    the exact trace is cheaper and has no variance.
    """
    shape = tuple(keras.ops.shape(x))
    if steps is None or shape[-1] <= steps:
        return None

    # Rademacher probes have lower variance than Gaussian probes
    signs = keras.random.randint((steps,) + shape, 0, 2, seed=seed)
    return keras.ops.cast(2 * signs - 1, keras.ops.dtype(x))


def _hutchinson(
    f: Callable,
    x: Tensor,
    steps: int = 1,
    return_output: bool = False,
    seed: int | keras.random.SeedGenerator = None,
    projectors: Tensor = None,
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

    if projectors is None:
        projectors = hutchinson_projectors(x, steps, seed=seed)
    steps = keras.ops.shape(projectors)[0]

    for step in range(steps):
        projector = projectors[step]
        trace += keras.ops.sum(vjp_fn(projector) * projector, axis=-1)

    trace = trace / steps

    if return_output:
        return fx, trace

    return trace
