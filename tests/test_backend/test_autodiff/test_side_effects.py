import keras
import pytest

from bayesflow._backend import grad, value_and_grad, jacfwd, jacrev, jvp, vjp, jit


def _make_counter_fn(counter, out_fn):
    def fn(x):
        counter["count"] += 1
        return out_fn(x)

    return fn


def test_grad_side_effect(jit_compile):
    counter = {"count": 0}
    x = keras.random.uniform(())
    fn = _make_counter_fn(counter, lambda x: keras.ops.sum(x))

    grad_fn = grad(fn)
    if jit_compile:
        grad_fn = jit(grad_fn)

    _ = grad_fn(x)
    # JAX executes the primal exactly once when computing grads for a Python function
    assert counter["count"] == 1


def test_value_and_grad_side_effect(jit_compile):
    counter = {"count": 0}
    x = keras.random.uniform(())
    fn = _make_counter_fn(counter, lambda x: keras.ops.sum(x))

    v_and_g = value_and_grad(fn)
    if jit_compile:
        v_and_g = jit(v_and_g)

    _ = v_and_g(x)
    assert counter["count"] == 1


def test_jacfwd_side_effect(jit_compile):
    counter = {"count": 0}
    x = keras.random.uniform((3,))
    fn = _make_counter_fn(counter, lambda x: keras.ops.sum(x))

    jac = jacfwd(fn)
    if jit_compile:
        jac = jit(jac)

    _ = jac(x)
    assert counter["count"] == 1


def test_jacrev_side_effect(jit_compile):
    if jit_compile and keras.backend.backend() == "torch":
        pytest.skip("torch's jacrev is not yet compatible with jit compilation.")

    counter = {"count": 0}
    x = keras.random.uniform((3,))
    fn = _make_counter_fn(counter, lambda x: keras.ops.sum(x))

    jac = jacrev(fn)
    if jit_compile:
        jac = jit(jac)

    _ = jac(x)
    assert counter["count"] == 1


def test_jvp_side_effect(jit_compile):
    counter = {"count": 0}
    x = keras.random.uniform(())
    tangent = keras.random.uniform(())
    fn = _make_counter_fn(counter, lambda x: keras.ops.sum(x))

    def call_jvp():
        return jvp(fn, [x], [tangent])

    # jvp is a function that takes already-created primals/tangents, so
    # ensure inputs are created outside any tracing to avoid RNG issues
    _ = jvp(fn, [x], [tangent])
    assert counter["count"] == 1

    if jit_compile:
        jf = jit(lambda: jvp(fn, [x], [tangent]))
        # calling jit-wrapped lambda should not re-run the primal more than once
        counter["count"] = 0
        _ = jf()
        assert counter["count"] == 1


def test_vjp_side_effect(jit_compile):
    if jit_compile and keras.backend.backend() == "torch":
        pytest.skip("torch's vjp is not yet compatible with jit compilation.")

    counter = {"count": 0}
    x = keras.random.uniform(())
    cotangent = keras.random.uniform(())
    fn = _make_counter_fn(counter, lambda x: keras.ops.sum(x))

    value, vjp_fn = vjp(fn, x)
    # primal evaluation during vjp construction should run fn once
    assert counter["count"] == 1

    # calling the returned vjp function should NOT re-execute the primal
    _ = vjp_fn(cotangent)
    assert counter["count"] == 1

    if jit_compile:
        # wrap the vjp creation+call in a lambda and jit it; inputs are pre-created
        counter["count"] = 0
        jf = jit(lambda: vjp(fn, x)[1](cotangent))
        _ = jf()
        # expect a single execution of the Python-level fn
        assert counter["count"] == 1
