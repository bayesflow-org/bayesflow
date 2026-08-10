import inspect

import keras
import numpy as np

from bayesflow.types import Tensor


def logits_relative_to_target(logits: Tensor, targets: Tensor) -> Tensor:
    """Express logits relative to the target model m = argmax(targets)."""
    m = keras.ops.cast(keras.ops.argmax(targets, axis=-1), dtype="int32")
    m_idx = keras.ops.expand_dims(m, axis=-1)
    logit_m = keras.ops.take_along_axis(logits, m_idx, axis=-1)
    return logits - logit_m


def resolve_seed(seed, seed_generator):
    """Resolve a user-provided ``seed`` against an instance's own seed generator.

    An integer is converted to a fresh ``SeedGenerator``, a ``SeedGenerator`` is passed through
    unchanged, and ``None`` falls back to ``seed_generator``.

    Every object that draws randomness of its own owns a ``self.seed_generator`` and passes it
    here, so such draws never fall through to Keras' global generator (which cannot be traced
    under ``jax.jit``). Objects that merely forward ``seed`` to a sub-component without drawing
    randomness themselves own no generator and do not call this function at all.
    """
    if isinstance(seed, int):
        return keras.random.SeedGenerator(seed)
    if seed is None:
        return seed_generator
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
