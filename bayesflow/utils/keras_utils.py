import inspect
from collections.abc import Mapping

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

    Every object that owns randomness passes its own ``self.seed_generator`` here, whether it
    draws itself or fans the seed out to sub-components. Draws therefore never fall through to
    Keras' global generator (which cannot be traced under ``jax.jit``), and an integer seed
    becomes one generator shared by all sub-components and batches of the call, rather than one
    identically seeded generator per draw site.
    """
    if isinstance(seed, int):
        return keras.random.SeedGenerator(seed)
    if seed is None:
        return seed_generator
    return seed


def multinomial_allocation(weights: Mapping[str, float], num_samples: int, seed=None) -> dict[str, int]:
    """Allocate `num_samples` draws across `weights` via multinomial sampling."""
    names = tuple(weights.keys())
    probs = np.array(list(weights.values()), dtype=keras.config.floatx())

    num_categories = len(probs)
    logits_broadcast = keras.ops.broadcast_to(
        keras.ops.expand_dims(keras.ops.log(probs), axis=0), (num_samples, num_categories)
    )
    cat_indices = keras.ops.squeeze(keras.random.categorical(logits_broadcast, num_samples=1, seed=seed), axis=-1)
    one_hot = keras.ops.one_hot(cat_indices, num_categories)
    counts = keras.ops.sum(one_hot, axis=0)

    return {name: int(count) for name, count in zip(names, counts)}


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
