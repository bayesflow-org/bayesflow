import keras
import numpy as np
from keras.ops import convert_to_numpy as to_np

from bayesflow.backend import jacrev, jacfwd


def test_jacrev():
    w = keras.random.normal((32, 16))
    b = keras.random.normal((32,))

    def fn(_x):
        return keras.ops.dot(w, _x) + b

    x = keras.random.normal((16,))
    jac = jacrev(fn)(x)

    assert keras.ops.is_tensor(jac)
    assert keras.ops.shape(jac) == keras.ops.shape(w)
    np.testing.assert_allclose(to_np(jac), to_np(w))


def test_jacrev_unary_scalar(fn_unary_scalar):
    x = keras.random.uniform(())
    jac_fn = jacrev(fn_unary_scalar)
    actual = jac_fn(x)

    assert keras.ops.is_tensor(actual)
    # For a scalar function of a scalar, the jacobian is a scalar
    assert keras.ops.shape(actual) == ()


def test_jacrev_unary_vector(fn_unary_vector):
    x = keras.random.uniform((2,))
    jac_fn = jacrev(fn_unary_vector)
    actual = jac_fn(x)

    assert keras.ops.is_tensor(actual)
    # For a vector function of a vector, the jacobian should be a vector
    assert keras.ops.shape(actual) == keras.ops.shape(x)


def test_jacrev_binary_scalars(fn_binary_scalars):
    x = keras.random.uniform(())
    y = keras.random.uniform(())

    # Test with single argnums
    jac_fn_x = jacrev(fn_binary_scalars, argnums=0)
    actual_x = jac_fn_x(x, y)
    assert keras.ops.is_tensor(actual_x)

    jac_fn_y = jacrev(fn_binary_scalars, argnums=1)
    actual_y = jac_fn_y(x, y)
    assert keras.ops.is_tensor(actual_y)

    # Test with multiple argnums
    jac_fn_xy = jacrev(fn_binary_scalars, argnums=(0, 1))
    actual_xy = jac_fn_xy(x, y)
    assert isinstance(actual_xy, tuple)
    assert len(actual_xy) == 2
    assert keras.ops.is_tensor(actual_xy[0])
    assert keras.ops.is_tensor(actual_xy[1])


def test_jacrev_binary_vectors(fn_binary_vectors):
    x = keras.random.uniform((2,))
    y = keras.random.uniform((2,))

    # Test with single argnums
    jac_fn_x = jacrev(fn_binary_vectors, argnums=0)
    actual_x = jac_fn_x(x, y)
    assert keras.ops.is_tensor(actual_x)

    jac_fn_y = jacrev(fn_binary_vectors, argnums=1)
    actual_y = jac_fn_y(x, y)
    assert keras.ops.is_tensor(actual_y)

    # Test with multiple argnums
    jac_fn_xy = jacrev(fn_binary_vectors, argnums=(0, 1))
    actual_xy = jac_fn_xy(x, y)
    assert isinstance(actual_xy, tuple)
    assert len(actual_xy) == 2
    assert keras.ops.is_tensor(actual_xy[0])
    assert keras.ops.is_tensor(actual_xy[1])


def test_jacrev_jacfwd_consistency(fn_unary_vector):
    x = keras.random.uniform((2,))

    jac_rev = jacrev(fn_unary_vector)(x)
    jac_fwd = jacfwd(fn_unary_vector)(x)

    assert keras.ops.shape(jac_rev) == keras.ops.shape(jac_fwd)
    np.testing.assert_allclose(to_np(jac_rev), to_np(jac_fwd), rtol=1e-5)


def test_jacrev_with_aux():
    w = keras.random.normal((4, 2))

    def fn_with_aux(x):
        y = keras.ops.dot(w, x)
        aux = keras.ops.sum(x)
        return y, aux

    x = keras.random.normal((2,))
    jac, aux = jacrev(fn_with_aux, has_aux=True)(x)

    assert keras.ops.is_tensor(jac)
    assert keras.ops.is_tensor(aux)
    assert keras.ops.shape(jac) == keras.ops.shape(w)


def test_jacrev_multiple_outputs():
    w = keras.random.normal((3, 2))

    def fn(x):
        return keras.ops.dot(w, x), keras.ops.sum(x)

    x = keras.random.normal((2,))
    jac = jacrev(fn)(x)

    assert isinstance(jac, tuple)
    assert len(jac) == 2
    assert keras.ops.is_tensor(jac[0])
    assert keras.ops.is_tensor(jac[1])
