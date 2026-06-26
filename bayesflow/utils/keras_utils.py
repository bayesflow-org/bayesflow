import inspect

import keras
import numpy as np

from bayesflow.types import Tensor


def resolve_seed(seed):
    """Convert an integer seed to a SeedGenerator; pass a SeedGenerator or None through unchanged."""
    if isinstance(seed, int):
        return keras.random.SeedGenerator(seed)
    return seed


def call_accepts_kwarg(call, key: str) -> bool:
    """Return whether a callable accepts a keyword argument.

    Parameters
    ----------
    call : Callable
        Callable to inspect.
    key : str
        Keyword argument name.

    Returns
    -------
    bool
        ``True`` if *call* explicitly accepts *key* or has ``**kwargs``.
    """
    try:
        parameters = inspect.signature(call).parameters
    except (TypeError, ValueError):
        return False

    return key in parameters or any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()
    )


def inverse_shifted_softplus(x: Tensor, shift: float = np.log(np.e - 1), beta: float = 1.0, threshold: float = 20.0):
    """Inverse of the shifted softplus function."""
    return inverse_softplus(x, beta=beta, threshold=threshold) - shift


def inverse_softplus(x: Tensor, beta: float = 1.0, threshold: float = 20.0) -> Tensor:
    """Numerically stabilized inverse softplus function."""
    return keras.ops.where(beta * x > threshold, x, keras.ops.log(keras.ops.expm1(beta * x)) / beta)


def shifted_softplus(x: Tensor, shift: float = np.log(np.e - 1)) -> Tensor:
    """Shifted version of the softplus function such that shifted_softplus(0) = 1"""
    return keras.ops.softplus(x + shift)
