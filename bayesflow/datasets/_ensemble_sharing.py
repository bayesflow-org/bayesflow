import numpy as np


def ring_starts(pool_size: int, ensemble_size: int) -> np.ndarray:
    """Deterministic start positions using rounding: round(m * pool_size / ensemble_size)."""
    if pool_size < 1:
        raise ValueError("pool_size must be >= 1.")
    if ensemble_size < 1:
        raise ValueError("ensemble_size must be >= 1.")
    starts = np.round(pool_size / ensemble_size * np.arange(ensemble_size)).astype("int64") % pool_size
    return starts


def ring_window_indices(pool_size: int, window_size: int, starts: np.ndarray) -> np.ndarray:
    """
    Return indices of shape (ensemble_size, window_size), where row m is a ring window starting at starts[m].
    """
    if window_size < 0:
        raise ValueError("window_size must be >= 0.")
    if window_size == 0:
        return np.zeros((len(starts), 0), dtype="int64")
    base = np.arange(window_size, dtype="int64")[None, :]
    idx = (starts[:, None] + base) % pool_size
    return idx
