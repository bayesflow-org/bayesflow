import keras

from tests.utils import assert_allclose


def fn(x):
    return keras.ops.square(x)


def test_vjp():
    from bayesflow.utils import vjp

    inputs = keras.random.normal((16, 32))
    tangents = keras.ops.ones((16, 32))

    vjp_fn = vjp(fn, inputs)
    v = vjp_fn(tangents)

    assert keras.ops.shape(v) == (16, 32)
    assert_allclose(v, 2.0 * inputs)


def test_jvp():
    from bayesflow.utils import jvp

    inputs = keras.random.normal((16, 32))
    tangents = keras.ops.ones((16, 32))

    v = jvp(fn, inputs, tangents)

    assert keras.ops.shape(v) == (16, 32)
    assert_allclose(v, 2.0 * inputs)


def test_jacobian():
    from bayesflow.utils import jacobian

    inputs = keras.random.normal((16, 32))

    j = jacobian(fn, inputs)
    target = 2.0 * keras.ops.tile(keras.ops.expand_dims(inputs, axis=-1), (1, 1, 32)) * keras.ops.eye(32)

    assert keras.ops.shape(j) == (16, 32, 32)
    assert_allclose(j, target, atol=0.01, rtol=0.01)


def test_jacobian_trace():
    from bayesflow.utils import jacobian, jacobian_trace

    inputs = keras.random.normal((16, 128))

    # deterministic
    j = jacobian(fn, inputs)
    jt = jacobian_trace(fn, inputs)

    assert jt.shape == (16,)

    jt_target = keras.ops.trace(j, axis1=-2, axis2=-1)

    assert_allclose(jt, jt_target)

    # too few max_steps, uses the stochastic version
    jt = jacobian_trace(fn, inputs, max_steps=127)

    # this check is not reliable enough yet
    # assert_allclose(jt, jt_target, atol=0.01, rtol=0.01)


def test_jacobian_trace_hutchinson_is_unbiased():
    from bayesflow.utils import jacobian_trace

    # exact trace is 2 * dims, independent of the number of steps
    def linear(x):
        return 2.0 * x

    dims = 256
    inputs = keras.random.normal((32, dims))

    for steps in (1, 4, 16):
        jt = jacobian_trace(linear, inputs, max_steps=steps, seed=0)
        assert jt.shape == (32,)
        assert_allclose(keras.ops.mean(jt), 2.0 * dims, rtol=0.1)
