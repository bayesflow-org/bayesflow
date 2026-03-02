import numpy as np
import pytest

from bayesflow.datasets._ensemble_sharing import ring_starts, ring_window_indices


def test_ring_starts_rejects_invalid_args():
    with pytest.raises(ValueError):
        ring_starts(pool_size=0, ensemble_size=1)
    with pytest.raises(ValueError):
        ring_starts(pool_size=1, ensemble_size=0)


def test_ring_starts_basic_properties_and_known_example():
    starts = ring_starts(pool_size=10, ensemble_size=6)
    assert starts.shape == (6,)
    assert np.all((0 <= starts) & (starts < 10))
    assert np.array_equal(starts, np.array([0, 2, 3, 5, 7, 8], dtype=np.int64))


def test_ring_window_indices_rejects_invalid_window_size():
    with pytest.raises(ValueError):
        ring_window_indices(pool_size=5, window_size=-1, starts=np.array([0, 1], dtype=np.int64))
    with pytest.raises(ValueError):
        ring_window_indices(pool_size=5, window_size=0, starts=np.array([0, 1], dtype=np.int64))


def test_ring_window_indices_wraps():
    starts = np.array([8, 9], dtype=np.int64)
    idx = ring_window_indices(pool_size=10, window_size=4, starts=starts)
    expected = np.array([[8, 9, 0, 1], [9, 0, 1, 2]], dtype=np.int64)
    assert np.array_equal(idx, expected)
