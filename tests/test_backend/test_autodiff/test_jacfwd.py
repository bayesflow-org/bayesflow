import keras
import numpy as np
from keras.ops import convert_to_numpy as to_np

from bayesflow._backend import jacfwd, jit


def test_jacfwd():
    w = keras.random.normal((32, 16))
    b = keras.random.normal((32,))

    def fn(_x):
        return keras.ops.dot(w, _x) + b

    x = keras.random.normal((16,))
    jac = jacfwd(fn)(x)

    assert keras.ops.is_tensor(jac)
    assert keras.ops.shape(jac) == keras.ops.shape(w)
    np.testing.assert_allclose(to_np(jac), to_np(w))


def test_jacfwd_unary_scalar(fn_unary_scalar, jit_compile):
    x = keras.random.uniform(())
    jac_fn = jacfwd(fn_unary_scalar)

    if jit_compile:
        jac_fn = jit(jac_fn)

    actual = jac_fn(x)

    assert keras.ops.is_tensor(actual)
    # For a scalar function of a scalar, the jacobian is a scalar
    assert keras.ops.shape(actual) == ()


def test_jacfwd_unary_vector(fn_unary_vector, jit_compile):
    x = keras.random.uniform((2,))
    jac_fn = jacfwd(fn_unary_vector)

    if jit_compile:
        jac_fn = jit(jac_fn)

    actual = jac_fn(x)

    assert keras.ops.is_tensor(actual)
    # For a vector function of a vector, the jacobian should be a vector
    assert keras.ops.shape(actual) == keras.ops.shape(x)


def test_jacfwd_binary_scalars(fn_binary_scalars, jit_compile):
    x = keras.random.uniform(())
    y = keras.random.uniform(())

    # Test with single argnums
    jac_fn = jacfwd(fn_binary_scalars, argnums=0)

    if jit_compile:
        jac_fn = jit(jac_fn)

    actual_x = jac_fn(x, y)
    assert keras.ops.is_tensor(actual_x)

    jac_fn = jacfwd(fn_binary_scalars, argnums=1)

    if jit_compile:
        jac_fn = jit(jac_fn)

    actual_y = jac_fn(x, y)
    assert keras.ops.is_tensor(actual_y)

    # Test with multiple argnums
    jac_fn = jacfwd(fn_binary_scalars, argnums=(0, 1))

    if jit_compile:
        jac_fn = jit(jac_fn)

    actual_xy = jac_fn(x, y)
    assert isinstance(actual_xy, tuple)
    assert len(actual_xy) == 2
    assert keras.ops.is_tensor(actual_xy[0])
    assert keras.ops.is_tensor(actual_xy[1])


def test_jacfwd_binary_vectors(fn_binary_vectors, jit_compile):
    x = keras.random.uniform((2,))
    y = keras.random.uniform((2,))

    # Test with single argnums
    jac_fn = jacfwd(fn_binary_vectors, argnums=0)

    if jit_compile:
        jac_fn = jit(jac_fn)

    actual_x = jac_fn(x, y)
    assert keras.ops.is_tensor(actual_x)

    jac_fn = jacfwd(fn_binary_vectors, argnums=1)

    if jit_compile:
        jac_fn = jit(jac_fn)

    actual_y = jac_fn(x, y)
    assert keras.ops.is_tensor(actual_y)

    # Test with multiple argnums
    jac_fn = jacfwd(fn_binary_vectors, argnums=(0, 1))

    if jit_compile:
        jac_fn = jit(jac_fn)

    actual_xy = jac_fn(x, y)
    assert isinstance(actual_xy, tuple)
    assert len(actual_xy) == 2
    assert keras.ops.is_tensor(actual_xy[0])
    assert keras.ops.is_tensor(actual_xy[1])


def test_jacfwd_with_aux(jit_compile):
    w = keras.random.normal((4, 2))

    def fn_with_aux(x):
        y = keras.ops.dot(w, x)
        aux = keras.ops.sum(x)
        return y, aux

    x = keras.random.normal((2,))
    jac_fn = jacfwd(fn_with_aux, has_aux=True)

    if jit_compile:
        jac_fn = jit(jac_fn)

    jac, aux = jac_fn(x)

    assert keras.ops.is_tensor(jac)
    assert keras.ops.is_tensor(aux)
    assert keras.ops.shape(jac) == keras.ops.shape(w)


def test_jacfwd_multiple_outputs(jit_compile):
    w = keras.random.normal((3, 2))

    def fn(x):
        return keras.ops.dot(w, x), keras.ops.sum(x)

    x = keras.random.normal((2,))
    jac_fn = jacfwd(fn)

    if jit_compile:
        jac_fn = jit(jac_fn)

    jac = jac_fn(x)

    assert isinstance(jac, tuple)
    assert len(jac) == 2
    assert keras.ops.is_tensor(jac[0])
    assert keras.ops.is_tensor(jac[1])
