import keras
import numpy as np
from keras.ops import convert_to_numpy as to_np

from bayesflow._backend import jacrev as jacobian, jit, value_and_grad


def test_value_and_grad_unary_scalar(fn_unary_scalar, jit_compile):
    x = keras.random.uniform(())
    grad_fn = value_and_grad(fn_unary_scalar)

    if jit_compile:
        grad_fn = jit(grad_fn)

    actual_value, actual_grad = grad_fn(x)
    expected_value = fn_unary_scalar(x)
    expected_grad = jacobian(fn_unary_scalar)(x)

    assert keras.ops.is_tensor(actual_value)
    assert keras.ops.is_tensor(expected_value)
    assert keras.ops.shape(actual_value) == keras.ops.shape(expected_value)
    np.testing.assert_allclose(to_np(actual_value), to_np(expected_value))

    assert keras.ops.is_tensor(actual_grad)
    assert keras.ops.is_tensor(expected_grad)
    assert keras.ops.shape(actual_grad) == keras.ops.shape(expected_grad)
    np.testing.assert_allclose(to_np(actual_grad), to_np(expected_grad))


def test_value_and_grad_unary_vector(fn_unary_vector, jit_compile):
    x = keras.random.uniform((2,))
    grad_fn = value_and_grad(fn_unary_vector)

    if jit_compile:
        grad_fn = jit(grad_fn)

    actual_value, actual_grad = grad_fn(x)
    expected_value = fn_unary_vector(x)
    expected_grad = jacobian(fn_unary_vector)(x)

    assert keras.ops.is_tensor(actual_value)
    assert keras.ops.is_tensor(expected_value)
    assert keras.ops.shape(actual_value) == keras.ops.shape(expected_value)
    np.testing.assert_allclose(to_np(actual_value), to_np(expected_value))

    assert keras.ops.is_tensor(actual_grad)
    assert keras.ops.is_tensor(expected_grad)
    assert keras.ops.shape(actual_grad) == keras.ops.shape(expected_grad)
    np.testing.assert_allclose(to_np(actual_grad), to_np(expected_grad))


def test_value_and_grad_binary_scalars(fn_binary_scalars, jit_compile):
    x = keras.random.uniform(())
    y = keras.random.uniform(())

    # Test with single argnums
    grad_fn = value_and_grad(fn_binary_scalars, argnums=0)

    if jit_compile:
        grad_fn = jit(grad_fn)

    actual_value_x, actual_grad_x = grad_fn(x, y)
    expected_value = fn_binary_scalars(x, y)
    expected_grad_x = jacobian(fn_binary_scalars, argnums=0)(x, y)

    assert keras.ops.is_tensor(actual_value_x)
    assert keras.ops.is_tensor(expected_value)
    np.testing.assert_allclose(to_np(actual_value_x), to_np(expected_value))
    assert keras.ops.is_tensor(actual_grad_x)
    assert keras.ops.shape(actual_grad_x) == keras.ops.shape(expected_grad_x)
    np.testing.assert_allclose(to_np(actual_grad_x), to_np(expected_grad_x))

    # Test with multiple argnums
    grad_fn = value_and_grad(fn_binary_scalars, argnums=(0, 1))

    if jit_compile:
        grad_fn = jit(grad_fn)

    actual_value, actual_grad = grad_fn(x, y)
    expected_grad = jacobian(fn_binary_scalars, argnums=(0, 1))(x, y)

    assert keras.ops.is_tensor(actual_value)
    assert keras.ops.is_tensor(expected_value)
    assert keras.ops.shape(actual_value) == keras.ops.shape(expected_value)
    np.testing.assert_allclose(to_np(actual_value), to_np(expected_value))

    assert isinstance(actual_grad, tuple)
    assert len(actual_grad) == 2
    assert keras.ops.is_tensor(actual_grad[0])
    assert keras.ops.is_tensor(actual_grad[1])
    assert keras.ops.shape(actual_grad[0]) == keras.ops.shape(expected_grad[0])
    assert keras.ops.shape(actual_grad[1]) == keras.ops.shape(expected_grad[1])
    np.testing.assert_allclose(to_np(actual_grad[0]), to_np(expected_grad[0]))
    np.testing.assert_allclose(to_np(actual_grad[1]), to_np(expected_grad[1]))


def test_value_and_grad_binary_vectors(fn_binary_vectors, jit_compile):
    x = keras.random.uniform((2,))
    y = keras.random.uniform((2,))

    # Test with single argnums
    grad_fn = value_and_grad(fn_binary_vectors, argnums=0)

    if jit_compile:
        grad_fn = jit(grad_fn)

    actual_value_x, actual_grad_x = grad_fn(x, y)
    expected_value = fn_binary_vectors(x, y)
    expected_grad_x = jacobian(fn_binary_vectors, argnums=0)(x, y)

    assert keras.ops.is_tensor(actual_value_x)
    assert keras.ops.is_tensor(expected_value)
    np.testing.assert_allclose(to_np(actual_value_x), to_np(expected_value))
    assert keras.ops.is_tensor(actual_grad_x)
    assert keras.ops.shape(actual_grad_x) == keras.ops.shape(expected_grad_x)
    np.testing.assert_allclose(to_np(actual_grad_x), to_np(expected_grad_x))

    # Test with multiple argnums
    grad_fn = value_and_grad(fn_binary_vectors, argnums=(0, 1))

    if jit_compile:
        grad_fn = jit(grad_fn)

    actual_value, actual_grad = grad_fn(x, y)
    expected_grad = jacobian(fn_binary_vectors, argnums=(0, 1))(x, y)

    assert keras.ops.is_tensor(actual_value)
    assert keras.ops.is_tensor(expected_value)
    assert keras.ops.shape(actual_value) == keras.ops.shape(expected_value)
    np.testing.assert_allclose(to_np(actual_value), to_np(expected_value))

    assert isinstance(actual_grad, tuple)
    assert len(actual_grad) == 2
    assert keras.ops.is_tensor(actual_grad[0])
    assert keras.ops.is_tensor(actual_grad[1])
    assert keras.ops.shape(actual_grad[0]) == keras.ops.shape(expected_grad[0])
    assert keras.ops.shape(actual_grad[1]) == keras.ops.shape(expected_grad[1])
    np.testing.assert_allclose(to_np(actual_grad[0]), to_np(expected_grad[0]))
    np.testing.assert_allclose(to_np(actual_grad[1]), to_np(expected_grad[1]))
