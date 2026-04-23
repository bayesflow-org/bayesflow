import keras
import numpy as np
from keras.ops import convert_to_numpy as to_np

from bayesflow.backend import jacrev as jacobian, grad


def test_grad_unary_scalar(fn_unary_scalar):
    x = keras.random.uniform(())
    grad_fn = grad(fn_unary_scalar)
    actual = grad_fn(x)
    expected = jacobian(fn_unary_scalar)(x)

    assert keras.ops.is_tensor(actual)
    assert keras.ops.is_tensor(expected)
    assert keras.ops.shape(actual) == keras.ops.shape(expected)
    np.testing.assert_allclose(to_np(actual), to_np(expected))


def test_grad_unary_vector(fn_unary_vector):
    x = keras.random.uniform((2,))
    grad_fn = grad(fn_unary_vector)
    actual = grad_fn(x)
    expected = jacobian(fn_unary_vector)(x)

    assert keras.ops.is_tensor(actual)
    assert keras.ops.is_tensor(expected)
    assert keras.ops.shape(actual) == keras.ops.shape(expected)
    np.testing.assert_allclose(to_np(actual), to_np(expected))


def test_grad_binary_scalars(fn_binary_scalars):
    x = keras.random.uniform(())
    y = keras.random.uniform(())

    # Test with single argnums
    grad_fn_x = grad(fn_binary_scalars, argnums=0)
    actual_x = grad_fn_x(x, y)
    expected_x = jacobian(fn_binary_scalars, argnums=0)(x, y)

    assert keras.ops.is_tensor(actual_x)
    assert keras.ops.is_tensor(expected_x)
    assert keras.ops.shape(actual_x) == keras.ops.shape(expected_x)
    np.testing.assert_allclose(to_np(actual_x), to_np(expected_x))

    # Test with multiple argnums
    grad_fn = grad(fn_binary_scalars, argnums=(0, 1))
    actual = grad_fn(x, y)
    expected = jacobian(fn_binary_scalars, argnums=(0, 1))(x, y)

    assert isinstance(actual, tuple)
    assert len(actual) == 2
    assert keras.ops.is_tensor(actual[0])
    assert keras.ops.is_tensor(actual[1])
    assert keras.ops.shape(actual[0]) == keras.ops.shape(expected[0])
    assert keras.ops.shape(actual[1]) == keras.ops.shape(expected[1])
    np.testing.assert_allclose(to_np(actual[0]), to_np(expected[0]))
    np.testing.assert_allclose(to_np(actual[1]), to_np(expected[1]))


def test_grad_binary_vectors(fn_binary_vectors):
    x = keras.random.uniform((2,))
    y = keras.random.uniform((2,))

    # Test with single argnums
    grad_fn_x = grad(fn_binary_vectors, argnums=0)
    actual_x = grad_fn_x(x, y)
    expected_x = jacobian(fn_binary_vectors, argnums=0)(x, y)

    assert keras.ops.is_tensor(actual_x)
    assert keras.ops.is_tensor(expected_x)
    assert keras.ops.shape(actual_x) == keras.ops.shape(expected_x)
    np.testing.assert_allclose(to_np(actual_x), to_np(expected_x))

    # Test with multiple argnums
    grad_fn = grad(fn_binary_vectors, argnums=(0, 1))
    actual = grad_fn(x, y)
    expected = jacobian(fn_binary_vectors, argnums=(0, 1))(x, y)

    assert isinstance(actual, tuple)
    assert len(actual) == 2
    assert keras.ops.is_tensor(actual[0])
    assert keras.ops.is_tensor(actual[1])
    assert keras.ops.shape(actual[0]) == keras.ops.shape(expected[0])
    assert keras.ops.shape(actual[1]) == keras.ops.shape(expected[1])
    np.testing.assert_allclose(to_np(actual[0]), to_np(expected[0]))
    np.testing.assert_allclose(to_np(actual[1]), to_np(expected[1]))
