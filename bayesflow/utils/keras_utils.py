from collections.abc import Sequence
import keras
import numpy as np

from bayesflow.types import Tensor


def resolve_seed(seed):
    """Convert an integer seed to a SeedGenerator; pass a SeedGenerator or None through unchanged."""
    if isinstance(seed, int):
        return keras.random.SeedGenerator(seed)
    return seed


def keras_multinomial(
    num_samples: int, probs: Sequence[float], *, seed: int | keras.random.SeedGenerator | None = None
):
    K = len(probs)
    logits_broadcast = keras.ops.broadcast_to(keras.ops.expand_dims(keras.ops.log(probs), axis=0), (num_samples, K))
    cat_indices = keras.ops.squeeze(keras.random.categorical(logits_broadcast, num_samples=1, seed=seed), axis=-1)
    one_hot = keras.ops.one_hot(cat_indices, K)
    counts = keras.ops.sum(one_hot, axis=0)

    return counts


def inverse_shifted_softplus(x: Tensor, shift: float = np.log(np.e - 1), beta: float = 1.0, threshold: float = 20.0):
    """Inverse of the shifted softplus function."""
    return inverse_softplus(x, beta=beta, threshold=threshold) - shift


def inverse_softplus(x: Tensor, beta: float = 1.0, threshold: float = 20.0) -> Tensor:
    """Numerically stabilized inverse softplus function."""
    return keras.ops.where(beta * x > threshold, x, keras.ops.log(keras.ops.expm1(beta * x)) / beta)


def shifted_softplus(x: Tensor, shift: float = np.log(np.e - 1)) -> Tensor:
    """Shifted version of the softplus function such that shifted_softplus(0) = 1"""
    return keras.ops.softplus(x + shift)
