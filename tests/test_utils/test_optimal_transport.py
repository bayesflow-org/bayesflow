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


@pytest.mark.parametrize(
    ["method", "partial", "condition_ratio"],
    [
        ("log_sinkhorn", True, 0.01),
        ("log_sinkhorn", False, 0.5),
        ("sinkhorn", True, 0.01),
        ("sinkhorn", False, 0.5),
    ],
)
def test_shapes(method, partial, condition_ratio):
    x = keras.random.normal((128, 8), seed=0)
    y = keras.random.normal((128, 8), seed=1)

    cond = None
    if condition_ratio < 0.5:
        cond = keras.random.normal((128, 4, 1), seed=2)

    ox, oy = optimal_transport(
        x,
        y,
        conditions=cond,
        regularization=1.0,
        seed=0,
        max_steps=10,
        method=method,
        partial=partial,
        condition_ratio=condition_ratio,
    )

    assert keras.ops.shape(ox) == keras.ops.shape(x)
    assert keras.ops.shape(oy) == keras.ops.shape(y)


@pytest.mark.parametrize(
    ["method", "partial", "condition_ratio"],
    [
        ("log_sinkhorn", True, 0.01),
        ("log_sinkhorn", False, 0.5),
        ("sinkhorn", True, 0.01),
        ("sinkhorn", False, 0.5),
    ],
)
def test_transport_cost_improves(method, partial, condition_ratio):
    x = keras.random.normal((128, 2), seed=0)
    y = keras.random.normal((128, 2), seed=1)

    cond = None
    if condition_ratio < 0.5:
        cond = keras.random.normal((128, 4, 1), seed=2)

    before_cost = keras.ops.sum(keras.ops.norm(x - y, axis=-1))

    x_after, y_after = optimal_transport(
        x,
        y,
        conditions=cond,
        regularization="auto",
        seed=0,
        max_steps=1000,
        method=method,
        partial=partial,
        condition_ratio=condition_ratio,
    )
    after_cost = keras.ops.sum(keras.ops.norm(x_after - y_after, axis=-1))

    assert after_cost < before_cost


@pytest.mark.parametrize(
    ["method", "partial"],
    [
        ("log_sinkhorn", True),
        ("log_sinkhorn", False),
        ("sinkhorn", True),
        ("sinkhorn", False),
    ],
)
def test_assignment_is_optimal(method, partial):
    y = keras.random.normal((16, 2), seed=0)
    p = keras.random.shuffle(keras.ops.arange(keras.ops.shape(y)[0]), seed=0)

    x = keras.ops.take(y, p, axis=0)

    _, _, assignments = optimal_transport(
        x, y, regularization="auto", seed=0, max_steps=10_000, method=method, return_assignments=True, partial=partial
    )

    # transport is stochastic, so it is expected that a small fraction of assignments does not match
    assert keras.ops.sum(assignments == p) > 14


def test_assignment_aligns_with_pot():
    try:
        from ot.bregman import sinkhorn_log
    except (ImportError, ModuleNotFoundError):
        pytest.skip("Need to install POT to run this test.")
        return

    x = keras.random.normal((16, 2), seed=0)
    p = keras.random.shuffle(keras.ops.arange(keras.ops.shape(x)[0]), seed=0)
    y = x[p]

    a = keras.ops.ones(keras.ops.shape(x)[0])
    b = keras.ops.ones(keras.ops.shape(y)[0])
    M = x[:, None] - y[None, :]
    M = keras.ops.norm(M, axis=-1)

    pot_plan = sinkhorn_log(a, b, M, numItermax=10_000, reg=1e-3, stopThr=1e-7)
    pot_assignments = keras.random.categorical(keras.ops.log(pot_plan), num_samples=1, seed=0)
    pot_assignments = keras.ops.squeeze(pot_assignments, axis=-1)

    _, _, assignments = optimal_transport(x, y, regularization=1e-3, seed=0, max_steps=10_000, return_assignments=True)

    assert_allclose(pot_assignments, assignments)


def test_sinkhorn_plan_correct_marginals():
    from bayesflow.utils.optimal_transport.sinkhorn import sinkhorn_plan

    x1 = keras.random.normal((10, 2), seed=0)
    x2 = keras.random.normal((20, 2), seed=1)

    assert keras.ops.all(keras.ops.isclose(keras.ops.sum(sinkhorn_plan(x1, x2), axis=0), 0.05, atol=1e-6))
    assert keras.ops.all(keras.ops.isclose(keras.ops.sum(sinkhorn_plan(x1, x2), axis=1), 0.1, atol=1e-6))


def test_sinkhorn_plan_aligns_with_pot():
    try:
        from ot.bregman import sinkhorn
    except (ImportError, ModuleNotFoundError):
        pytest.skip("Need to install POT to run this test.")

    from bayesflow.utils.optimal_transport.sinkhorn import sinkhorn_plan
    from bayesflow.utils.optimal_transport.ot_utils import euclidean

    x1 = keras.random.normal((10, 3), seed=0)
    x2 = keras.random.normal((20, 3), seed=1)

    a = keras.ops.ones(10) / 10
    b = keras.ops.ones(20) / 20
    M = euclidean(x1, x2)

    pot_result = sinkhorn(a, b, M, 0.1, stopThr=1e-8)
    our_result = sinkhorn_plan(x1, x2, regularization=0.1, rtol=1e-7)

    assert_allclose(pot_result, our_result)


def test_sinkhorn_plan_matches_analytical_result():
    from bayesflow.utils.optimal_transport.sinkhorn import sinkhorn_plan

    x1 = keras.ops.ones(16)
    x2 = keras.ops.ones(64)

    marginal_x1 = keras.ops.ones(16) / 16
    marginal_x2 = keras.ops.ones(64) / 64

    result = sinkhorn_plan(x1, x2, regularization=0.1)

    # If x1 and x2 are identical, the optimal plan is simply the outer product of the marginals
    expected = keras.ops.outer(marginal_x1, marginal_x2)

    assert_allclose(result, expected)


def test_log_sinkhorn_plan_correct_marginals():
    from bayesflow.utils.optimal_transport.log_sinkhorn import log_sinkhorn_plan

    x1 = keras.random.normal((10, 2), seed=0)
    x2 = keras.random.normal((20, 2), seed=1)

    assert keras.ops.all(
        keras.ops.isclose(keras.ops.logsumexp(log_sinkhorn_plan(x1, x2), axis=0), -keras.ops.log(20), atol=1e-3)
    )
    assert keras.ops.all(
        keras.ops.isclose(keras.ops.logsumexp(log_sinkhorn_plan(x1, x2), axis=1), -keras.ops.log(10), atol=1e-3)
    )


def test_log_sinkhorn_plan_aligns_with_pot():
    try:
        from ot.bregman import sinkhorn_log
    except (ImportError, ModuleNotFoundError):
        pytest.skip("Need to install POT to run this test.")

    from bayesflow.utils.optimal_transport.log_sinkhorn import log_sinkhorn_plan
    from bayesflow.utils.optimal_transport.ot_utils import euclidean

    x1 = keras.random.normal((100, 3), seed=0)
    x2 = keras.random.normal((200, 3), seed=1)

    a = keras.ops.ones(100) / 100
    b = keras.ops.ones(200) / 200
    M = euclidean(x1, x2)

    pot_result = keras.ops.log(sinkhorn_log(a, b, M, 0.1, stopThr=1e-7))  # sinkhorn_log returns probabilities
    our_result = log_sinkhorn_plan(x1, x2, regularization=0.1)

    assert_allclose(pot_result, our_result)


def test_log_sinkhorn_plan_matches_analytical_result():
    from bayesflow.utils.optimal_transport.log_sinkhorn import log_sinkhorn_plan

    x1 = keras.ops.ones(16)
    x2 = keras.ops.ones(64)

    marginal_x1 = keras.ops.ones(16) / 16
    marginal_x2 = keras.ops.ones(64) / 64

    result = keras.ops.exp(log_sinkhorn_plan(x1, x2, regularization=0.1))

    # If x1 and x2 are identical, the optimal plan is simply the outer product of the marginals
    expected = keras.ops.outer(marginal_x1, marginal_x2)

    assert_allclose(result, expected)
