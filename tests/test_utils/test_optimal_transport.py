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


def test_shapes():
    x = keras.random.normal((128, 8), seed=0)
    y = keras.random.normal((128, 8), seed=1)

    ox, oy = optimal_transport(x, y, regularization=1.0, seed=0, max_steps=10)

    assert keras.ops.shape(ox) == keras.ops.shape(x)
    assert keras.ops.shape(oy) == keras.ops.shape(y)


def test_transport_cost_improves():
    x = keras.random.normal((32, 8), seed=0)
    y = keras.random.normal((32, 8), seed=1)

    before_cost = keras.ops.sum(keras.ops.norm(x - y, axis=-1))

    x, y = optimal_transport(x, y, regularization=0.1, seed=0, max_steps=None)

    after_cost = keras.ops.sum(keras.ops.norm(x - y, axis=-1))

    assert after_cost < before_cost


def test_assignment_is_optimal():
    x = keras.ops.stack([keras.ops.linspace(-1, 1, 10), keras.ops.linspace(-1, 1, 10)])
    y = keras.ops.copy(x)

    # we could shuffle x and y, but flipping is a more reliable permutation
    y = keras.ops.flip(y, axis=0)

    x, y = optimal_transport(x, y, regularization=1e-3, seed=0, max_steps=1000)

    cost = keras.ops.sum(keras.ops.norm(x - y, axis=-1))

    assert_allclose(cost, 0.0)
