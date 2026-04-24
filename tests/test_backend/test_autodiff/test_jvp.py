import keras
import numpy as np
from keras.ops import convert_to_numpy as to_np

from bayesflow._backend import jacrev as jacobian, jvp


def test_jvp_unary_scalar(fn_unary_scalar):
    x = keras.random.uniform(())
    tangent = keras.random.uniform(())
    actual_value, actual_jvp = jvp(fn_unary_scalar, [x], [tangent])
    expected_value = fn_unary_scalar(x)
    expected_jvp = jacobian(fn_unary_scalar)(x) * tangent

    assert keras.ops.is_tensor(actual_value)
    assert keras.ops.is_tensor(expected_value)
    assert keras.ops.shape(actual_value) == keras.ops.shape(expected_value)
    np.testing.assert_allclose(to_np(actual_value), to_np(expected_value))

    assert keras.ops.is_tensor(actual_jvp)
    assert keras.ops.is_tensor(expected_jvp)
    assert keras.ops.shape(actual_jvp) == keras.ops.shape(expected_jvp)
    np.testing.assert_allclose(to_np(actual_jvp), to_np(expected_jvp))


def test_jvp_unary_vector(fn_unary_vector):
    x = keras.random.uniform((2,))
    tangent = keras.random.uniform((2,))
    actual_value, actual_jvp = jvp(fn_unary_vector, [x], [tangent])
    expected_value = fn_unary_vector(x)
    expected_jvp = keras.ops.dot(jacobian(fn_unary_vector)(x), tangent)

    assert keras.ops.is_tensor(actual_value)
    assert keras.ops.is_tensor(expected_value)
    assert keras.ops.shape(actual_value) == keras.ops.shape(expected_value)
    np.testing.assert_allclose(to_np(actual_value), to_np(expected_value))

    assert keras.ops.is_tensor(actual_jvp)
    assert keras.ops.is_tensor(expected_jvp)
    assert keras.ops.shape(actual_jvp) == keras.ops.shape(expected_jvp)
    np.testing.assert_allclose(to_np(actual_jvp), to_np(expected_jvp))


def test_jvp_binary_scalars(fn_binary_scalars):
    primals = [keras.random.uniform(()), keras.random.uniform(())]
    tangents = [keras.random.uniform(()), keras.random.uniform(())]

    # Test with all primals/tangents
    actual_value, actual_jvp = jvp(fn_binary_scalars, primals, tangents)
    expected_value = fn_binary_scalars(*primals)

    expected_jvp = keras.ops.dot(
        keras.ops.stack(jacobian(fn_binary_scalars, argnums=(0, 1))(*primals)), keras.ops.stack(tangents)
    )

    assert keras.ops.is_tensor(actual_value)
    assert keras.ops.is_tensor(expected_value)
    assert keras.ops.shape(actual_value) == keras.ops.shape(expected_value)
    np.testing.assert_allclose(to_np(actual_value), to_np(expected_value))

    assert keras.ops.is_tensor(actual_jvp)
    assert keras.ops.is_tensor(expected_jvp)
    assert keras.ops.shape(actual_jvp) == keras.ops.shape(expected_jvp)
    np.testing.assert_allclose(to_np(actual_jvp), to_np(expected_jvp))


def test_jvp_binary_vectors(fn_binary_vectors):
    primals = [keras.random.uniform((2,)), keras.random.uniform((2,))]
    tangents = [keras.random.uniform((2,)), keras.random.uniform((2,))]

    # Test with all primals/tangents
    actual_value, actual_jvp = jvp(fn_binary_vectors, primals, tangents)
    expected_value = fn_binary_vectors(*primals)

    expected_jvp = keras.ops.dot(
        keras.ops.concatenate(jacobian(fn_binary_vectors, argnums=(0, 1))(*primals)), keras.ops.concatenate(tangents)
    )

    assert keras.ops.is_tensor(actual_value)
    assert keras.ops.is_tensor(expected_value)
    assert keras.ops.shape(actual_value) == keras.ops.shape(expected_value)
    np.testing.assert_allclose(to_np(actual_value), to_np(expected_value))

    assert keras.ops.is_tensor(actual_jvp)
    assert keras.ops.is_tensor(expected_jvp)
    assert keras.ops.shape(actual_jvp) == keras.ops.shape(expected_jvp)
    np.testing.assert_allclose(to_np(actual_jvp), to_np(expected_jvp))
