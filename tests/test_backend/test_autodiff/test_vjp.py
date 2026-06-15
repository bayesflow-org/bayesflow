import keras
import numpy as np
import pytest
from keras.ops import convert_to_numpy as to_np

from bayesflow._backend import jacrev as jacobian, vjp, jit


def test_vjp_unary_scalar(fn_unary_scalar, jit_compile):
    if jit_compile and keras.backend.backend() == "torch":
        pytest.skip("torch's vjp is not yet compatible with jit compilation.")

    x = keras.random.uniform(())
    cotangent = keras.random.uniform(())
    actual_value, vjp_fn = vjp(fn_unary_scalar, x)

    if jit_compile:
        vjp_fn = jit(vjp_fn)

    out = vjp_fn(cotangent)

    assert isinstance(out, tuple)
    assert len(out) == 1
    actual_vjp = out[0]

    expected_value = fn_unary_scalar(x)
    expected_vjp = cotangent * jacobian(fn_unary_scalar)(x)

    assert keras.ops.is_tensor(actual_value)
    assert keras.ops.shape(actual_value) == keras.ops.shape(expected_value)
    np.testing.assert_allclose(to_np(actual_value), to_np(expected_value))

    assert keras.ops.is_tensor(actual_vjp)
    assert keras.ops.is_tensor(expected_vjp)
    assert keras.ops.shape(actual_vjp) == keras.ops.shape(expected_vjp)
    np.testing.assert_allclose(to_np(actual_vjp), to_np(expected_vjp))


def test_vjp_unary_vector(fn_unary_vector, jit_compile):
    if jit_compile and keras.backend.backend() == "torch":
        pytest.skip("torch's vjp is not yet compatible with jit compilation.")

    x = keras.random.uniform((2,))
    cotangent = keras.random.uniform(())
    actual_value, vjp_fn = vjp(fn_unary_vector, x)

    if jit_compile:
        vjp_fn = jit(vjp_fn)

    out = vjp_fn(cotangent)

    assert isinstance(out, tuple)
    assert len(out) == 1
    actual_vjp = out[0]

    expected_value = fn_unary_vector(x)
    expected_vjp = cotangent * jacobian(fn_unary_vector)(x)

    assert keras.ops.is_tensor(actual_value)
    assert keras.ops.is_tensor(expected_value)
    assert keras.ops.shape(actual_value) == keras.ops.shape(expected_value)
    np.testing.assert_allclose(to_np(actual_value), to_np(expected_value))

    assert keras.ops.is_tensor(actual_vjp)
    assert keras.ops.is_tensor(expected_vjp)
    assert keras.ops.shape(actual_vjp) == keras.ops.shape(expected_vjp)
    np.testing.assert_allclose(to_np(actual_vjp), to_np(expected_vjp))


def test_vjp_binary_scalars(fn_binary_scalars, jit_compile):
    if jit_compile and keras.backend.backend() == "torch":
        pytest.skip("torch's vjp is not yet compatible with jit compilation.")

    primals = [keras.random.uniform(()), keras.random.uniform(())]
    cotangent = keras.random.uniform(())
    actual_value, vjp_fn = vjp(fn_binary_scalars, *primals)

    if jit_compile:
        vjp_fn = jit(vjp_fn)

    actual_vjps = vjp_fn(cotangent)

    assert isinstance(actual_vjps, tuple)
    assert len(actual_vjps) == 2

    expected_value = fn_binary_scalars(*primals)

    _jacs = jacobian(fn_binary_scalars, argnums=(0, 1))(*primals)
    expected_vjps = [cotangent * j for j in _jacs]

    assert keras.ops.is_tensor(actual_value)
    assert keras.ops.is_tensor(expected_value)
    assert keras.ops.shape(actual_value) == keras.ops.shape(expected_value)
    np.testing.assert_allclose(to_np(actual_value), to_np(expected_value))

    for i in range(len(expected_vjps)):
        assert keras.ops.is_tensor(actual_vjps[i])
        assert keras.ops.is_tensor(expected_vjps[i])
        assert keras.ops.shape(actual_vjps[i]) == keras.ops.shape(expected_vjps[i])
        np.testing.assert_allclose(to_np(actual_vjps[i]), to_np(expected_vjps[i]))


def test_vjp_binary_vectors(fn_binary_vectors, jit_compile):
    if jit_compile and keras.backend.backend() == "torch":
        pytest.skip("torch's vjp is not yet compatible with jit compilation.")

    primals = [keras.random.uniform((2,)), keras.random.uniform((2,))]
    cotangent = keras.random.uniform(())
    actual_value, vjp_fn = vjp(fn_binary_vectors, *primals)

    if jit_compile:
        vjp_fn = jit(vjp_fn)

    actual_vjps = vjp_fn(cotangent)

    assert isinstance(actual_vjps, tuple)
    assert len(actual_vjps) == 2

    expected_value = fn_binary_vectors(*primals)

    _jacs = jacobian(fn_binary_vectors, argnums=(0, 1))(*primals)
    expected_vjps = [cotangent * j for j in _jacs]

    assert keras.ops.is_tensor(actual_value)
    assert keras.ops.is_tensor(expected_value)
    assert keras.ops.shape(actual_value) == keras.ops.shape(expected_value)
    np.testing.assert_allclose(to_np(actual_value), to_np(expected_value))

    for i in range(len(expected_vjps)):
        assert keras.ops.is_tensor(actual_vjps[i])
        assert keras.ops.is_tensor(expected_vjps[i])
        assert keras.ops.shape(actual_vjps[i]) == keras.ops.shape(expected_vjps[i])
        np.testing.assert_allclose(to_np(actual_vjps[i]), to_np(expected_vjps[i]))


def test_vjp_with_aux(jit_compile):
    if jit_compile and keras.backend.backend() == "torch":
        pytest.skip("torch's vjp is not yet compatible with jit compilation.")

    x = keras.random.uniform((2,))
    cotangent = keras.random.uniform(())

    def fn_with_aux(x):
        aux = keras.ops.sum(x)
        return keras.ops.sum(keras.ops.square(x)), aux

    actual_value, vjp_fn, actual_aux = vjp(fn_with_aux, x, has_aux=True)

    if jit_compile:
        vjp_fn = jit(vjp_fn)

    out = vjp_fn(cotangent)

    assert isinstance(out, tuple)
    assert len(out) == 1
    actual_vjp = out[0]

    expected_value = keras.ops.sum(keras.ops.square(x))
    expected_aux = keras.ops.sum(x)
    # grad of sum(x^2) = 2x, VJP = cotangent * 2x
    expected_vjp = cotangent * 2 * x

    assert keras.ops.is_tensor(actual_value)
    assert keras.ops.shape(actual_value) == keras.ops.shape(expected_value)
    np.testing.assert_allclose(to_np(actual_value), to_np(expected_value), rtol=1e-5)

    assert keras.ops.is_tensor(actual_aux)
    assert keras.ops.shape(actual_aux) == keras.ops.shape(expected_aux)
    np.testing.assert_allclose(to_np(actual_aux), to_np(expected_aux), rtol=1e-5)

    assert keras.ops.is_tensor(actual_vjp)
    assert keras.ops.shape(actual_vjp) == keras.ops.shape(expected_vjp)
    np.testing.assert_allclose(to_np(actual_vjp), to_np(expected_vjp), rtol=1e-5)
