import keras
import numpy as np
from keras.ops import convert_to_numpy as to_np

from bayesflow._backend import jit


def test_jit_basic():
    """jit-compiled function produces the same output as the original."""
    w = keras.random.normal((4, 2))

    def fn(x):
        return keras.ops.dot(w, x)

    x = keras.random.normal((2,))
    expected = fn(x)
    compiled = jit(fn)
    actual = compiled(x)

    assert keras.ops.is_tensor(actual)
    assert keras.ops.shape(actual) == keras.ops.shape(expected)
    np.testing.assert_allclose(to_np(actual), to_np(expected), rtol=1e-5)


def test_jit_idempotent():
    """Calling jit twice should still produce correct results."""

    def fn(x):
        return keras.ops.square(x)

    x = keras.random.normal((3,))
    expected = fn(x)
    compiled = jit(jit(fn))
    actual = compiled(x)

    assert keras.ops.is_tensor(actual)
    np.testing.assert_allclose(to_np(actual), to_np(expected), rtol=1e-5)


def test_jit_multiple_args():
    """jit works correctly with multiple arguments."""

    def fn(x, y):
        return keras.ops.dot(x, y)

    x = keras.random.normal((4,))
    y = keras.random.normal((4,))
    expected = fn(x, y)
    compiled = jit(fn)
    actual = compiled(x, y)

    assert keras.ops.is_tensor(actual)
    assert keras.ops.shape(actual) == keras.ops.shape(expected)
    np.testing.assert_allclose(to_np(actual), to_np(expected), rtol=1e-5)


def test_jit_returns_callable():
    """jit returns a callable."""

    def fn(x):
        return x

    compiled = jit(fn)
    assert callable(compiled)
