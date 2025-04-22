import keras
import pytest

from bayesflow.utils import optimal_transport
from tests.utils import assert_allclose


@pytest.mark.jax
def test_jit_compile():
    import jax

    x = keras.random.normal((128, 8), seed=0)
    y = keras.random.normal((128, 8), seed=1)

    ot = jax.jit(optimal_transport, static_argnames=["regularization", "seed"])
    ot(x, y, regularization=1.0, seed=0, max_steps=10)


@pytest.mark.parametrize("method", ["log_sinkhorn", "sinkhorn"])
def test_shapes(method):
    x = keras.random.normal((128, 8), seed=0)
    y = keras.random.normal((128, 8), seed=1)

    ox, oy = optimal_transport(x, y, regularization=1.0, seed=0, max_steps=10, method=method)

    assert keras.ops.shape(ox) == keras.ops.shape(x)
    assert keras.ops.shape(oy) == keras.ops.shape(y)


def test_transport_cost_improves():
    x = keras.random.normal((128, 2), seed=0)
    y = keras.random.normal((128, 2), seed=1)

    before_cost = keras.ops.sum(keras.ops.norm(x - y, axis=-1))

    x, y = optimal_transport(x, y, regularization=0.1, seed=0, max_steps=1000)

    after_cost = keras.ops.sum(keras.ops.norm(x - y, axis=-1))

    assert after_cost < before_cost


@pytest.mark.skip(reason="too unreliable")
def test_assignment_is_optimal():
    x = keras.random.normal((16, 2), seed=0)
    p = keras.random.shuffle(keras.ops.arange(keras.ops.shape(x)[0]), seed=0)
    optimal_assignments = keras.ops.argsort(p)

    y = x[p]

    x, y, assignments = optimal_transport(x, y, regularization=0.1, seed=0, max_steps=10_000, return_assignments=True)

    assert_allclose(assignments, optimal_assignments)


def test_assignment_aligns_with_pot():
    try:
        from ot.bregman import sinkhorn_log
    except (ImportError, ModuleNotFoundError):
        pytest.skip("Need to install POT to run this test.")

    x = keras.random.normal((16, 2), seed=0)
    p = keras.random.shuffle(keras.ops.arange(keras.ops.shape(x)[0]), seed=0)
    y = x[p]

    a = keras.ops.ones(keras.ops.shape(x)[0])
    b = keras.ops.ones(keras.ops.shape(y)[0])
    M = x[:, None] - y[None, :]
    M = keras.ops.norm(M, axis=-1)

    pot_plan = sinkhorn_log(a, b, M, reg=1e-3, numItermax=10_000, stopThr=1e-99)
    pot_assignments = keras.random.categorical(pot_plan, num_samples=1, seed=0)
    pot_assignments = keras.ops.squeeze(pot_assignments, axis=-1)

    _, _, assignments = optimal_transport(x, y, regularization=1e-3, seed=0, max_steps=10_000, return_assignments=True)

    assert_allclose(pot_assignments, assignments)
