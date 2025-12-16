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
    ["method", "partial_s", "condition_ratio"],
    [
        ("log_sinkhorn", 1.0, 0.01),
        ("log_sinkhorn", 0.8, 0.5),
        ("sinkhorn", 1.0, 0.01),
        ("sinkhorn", 0.8, 0.5),
    ],
)
def test_shapes(method, partial_s, condition_ratio):
    x = keras.random.normal((128, 8), seed=0)
    y = keras.random.normal((128, 8), seed=1)

    cond = None
    if condition_ratio < 0.5:
        cond = keras.random.normal((128, 4, 1), seed=2)

    ox, oy, ocond = optimal_transport(
        x,
        y,
        conditions=cond,
        regularization=1.0,
        seed=0,
        max_steps=10,
        method=method,
        partial_s=partial_s,
        condition_ratio=condition_ratio,
    )

    assert keras.ops.shape(ox) == keras.ops.shape(x)
    assert keras.ops.shape(oy) == keras.ops.shape(y)
    if cond is not None:
        assert keras.ops.shape(ocond) == keras.ops.shape(cond)


@pytest.mark.parametrize(
    ["method", "partial_s", "condition_ratio"],
    [
        ("log_sinkhorn", 1.0, 0.01),
        ("log_sinkhorn", 0.8, 0.5),
        ("sinkhorn", 1.0, 0.01),
        ("sinkhorn", 0.8, 0.5),
    ],
)
def test_transport_cost_improves(method, partial_s, condition_ratio):
    x = keras.random.normal((128, 2), seed=0)
    y = keras.random.normal((128, 2), seed=1)

    cond = None
    if condition_ratio < 0.5:
        cond = keras.random.normal((128, 4, 1), seed=2)

    before_cost = keras.ops.sum(keras.ops.norm(x - y, axis=-1))

    x_after, y_after, cond_after = optimal_transport(
        x,
        y,
        conditions=cond,
        regularization="auto",
        seed=0,
        max_steps=1000,
        method=method,
        partial_s=partial_s,
        condition_ratio=condition_ratio,
    )
    after_cost = keras.ops.sum(keras.ops.norm(x_after - y_after, axis=-1))

    assert after_cost < before_cost


@pytest.mark.parametrize(
    ["method", "partial_s"],
    [
        ("log_sinkhorn", 1.0),
        ("log_sinkhorn", 0.8),
        ("sinkhorn", 1.0),
        ("sinkhorn", 0.8),
    ],
)
def test_assignment_is_optimal(method, partial_s):
    y = keras.random.normal((16, 2), seed=0)
    p = keras.random.shuffle(keras.ops.arange(keras.ops.shape(y)[0]), seed=0)
    x = keras.ops.take(y, p, axis=0)

    _, _, _, assignments = optimal_transport(
        x,
        y,
        regularization="auto",
        seed=0,
        max_steps=10_000,
        method=method,
        return_assignments=True,
        partial_s=partial_s,
    )

    # transport is stochastic, so it is expected that a small fraction of assignments does not match
    assert keras.ops.sum(assignments == p) > 14


@pytest.mark.parametrize("method", ["log_sinkhorn", "sinkhorn"])
def test_no_nans_or_infs(method):
    """Test that algorithm produces finite values even with challenging inputs and auto regularization."""
    # Test with well-separated distributions
    x = keras.random.normal((64, 4), seed=0) * 10.0
    y = keras.random.normal((64, 4), seed=1) * 10.0 + 100.0

    ox, oy, _, assignments = optimal_transport(
        x, y, regularization="auto", seed=0, max_steps=1000, method=method, return_assignments=True
    )

    assert keras.ops.all(keras.ops.isfinite(ox))
    assert keras.ops.all(keras.ops.isfinite(oy))
    assert keras.ops.all(keras.ops.isfinite(assignments))


def test_assignment_aligns_with_pot():
    try:
        from ot.bregman import sinkhorn_log
    except (ImportError, ModuleNotFoundError):
        pytest.skip("Need to install POT to run this test.")
        return

    x = keras.random.normal((16, 2), seed=0)
    p = keras.random.shuffle(keras.ops.arange(keras.ops.shape(x)[0]), seed=0)
    y = keras.ops.take(x, p, axis=0)

    a = keras.ops.ones(keras.ops.shape(x)[0])
    b = keras.ops.ones(keras.ops.shape(y)[0])
    M = x[:, None] - y[None, :]
    M = keras.ops.norm(M, axis=-1)

    pot_plan = sinkhorn_log(a, b, M, numItermax=10_000, reg=1e-3, stopThr=1e-7)
    pot_assignments = keras.random.categorical(keras.ops.log(pot_plan), num_samples=1, seed=0)
    pot_assignments = keras.ops.squeeze(pot_assignments, axis=-1)

    _, _, _, assignments = optimal_transport(
        x, y, regularization=1e-3, seed=0, max_steps=10_000, return_assignments=True
    )

    assert_allclose(pot_assignments, assignments)


def test_sinkhorn_plan_correct_marginals():
    from bayesflow.utils.optimal_transport.sinkhorn import sinkhorn_plan

    x1 = keras.random.normal((10, 2), seed=0)
    x2 = keras.random.normal((20, 2), seed=1)

    plan = sinkhorn_plan(x1, x2, rtol=1e-7, max_steps=1000)

    assert keras.ops.all(keras.ops.isclose(keras.ops.sum(plan, axis=0), 0.05, atol=1e-6))
    assert keras.ops.all(keras.ops.isclose(keras.ops.sum(plan, axis=1), 0.1, atol=1e-6))


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

    pot_result = sinkhorn(a.numpy(), b.numpy(), M.numpy(), 0.1, stopThr=1e-8)
    our_result = sinkhorn_plan(x1, x2, regularization=0.1, rtol=1e-7)

    assert_allclose(pot_result, our_result)


@pytest.mark.parametrize("reg", ["auto", 0.1, 1.0])
def test_sinkhorn_plan_matches_analytical_result(reg):
    from bayesflow.utils.optimal_transport.sinkhorn import sinkhorn_plan

    x1 = keras.ops.ones(16)
    x2 = keras.ops.ones(64)

    marginal_x1 = keras.ops.ones(16) / 16
    marginal_x2 = keras.ops.ones(64) / 64

    result = sinkhorn_plan(x1, x2, regularization=reg)

    # If x1 and x2 are identical, the optimal plan is simply the outer product of the marginals
    expected = keras.ops.outer(marginal_x1, marginal_x2)

    assert_allclose(result, expected, rtol=1e-4)


def test_log_sinkhorn_plan_correct_marginals():
    from bayesflow.utils.optimal_transport.log_sinkhorn import log_sinkhorn_plan

    x1 = keras.random.normal((10, 2), seed=0)
    x2 = keras.random.normal((20, 2), seed=1)

    log_plan = log_sinkhorn_plan(x1, x2, rtol=1e-6, max_steps=1000)

    assert keras.ops.all(keras.ops.isclose(keras.ops.logsumexp(log_plan, axis=0), -keras.ops.log(20.0), atol=1e-3))
    assert keras.ops.all(keras.ops.isclose(keras.ops.logsumexp(log_plan, axis=1), -keras.ops.log(10.0), atol=1e-3))


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
    our_result = log_sinkhorn_plan(x1, x2, regularization=0.1, rtol=1e-6)

    assert_allclose(pot_result, our_result, rtol=1e-4)


def test_log_sinkhorn_plan_matches_analytical_result():
    from bayesflow.utils.optimal_transport.log_sinkhorn import log_sinkhorn_plan

    x1 = keras.ops.ones(16)
    x2 = keras.ops.ones(64)

    marginal_x1 = keras.ops.ones(16) / 16
    marginal_x2 = keras.ops.ones(64) / 64

    result = keras.ops.exp(log_sinkhorn_plan(x1, x2, regularization=0.1))

    # If x1 and x2 are identical, the optimal plan is simply the outer product of the marginals
    expected = keras.ops.outer(marginal_x1, marginal_x2)

    assert_allclose(result, expected, rtol=1e-4)


def test_sinkhorn_vs_log_sinkhorn_consistency():
    """Test that Sinkhorn and log-Sinkhorn produce consistent results."""
    from bayesflow.utils.optimal_transport.sinkhorn import sinkhorn_plan
    from bayesflow.utils.optimal_transport.log_sinkhorn import log_sinkhorn_plan

    x1 = keras.random.normal((20, 3), seed=0)
    x2 = keras.random.normal((30, 3), seed=1)

    plan_sinkhorn = sinkhorn_plan(x1, x2, regularization=0.1, rtol=1e-6)
    plan_log_sinkhorn = keras.ops.exp(log_sinkhorn_plan(x1, x2, regularization=0.1, rtol=1e-6))

    assert_allclose(plan_sinkhorn, plan_log_sinkhorn, rtol=1e-3)


@pytest.mark.parametrize(
    ["method", "s"],
    [
        ("log_sinkhorn", 0.3),
        ("log_sinkhorn", 0.8),
        ("sinkhorn", 0.4),
        ("sinkhorn", 0.7),
    ],
)
def test_partial_ot_leaves_unmatched_mass(method, s):
    """Test that partial OT correctly leaves a fraction of mass unmatched."""
    if method == "sinkhorn":
        from bayesflow.utils.optimal_transport.sinkhorn import sinkhorn_plan as sinkhorn
    else:
        from bayesflow.utils.optimal_transport.log_sinkhorn import log_sinkhorn_plan as sinkhorn
    n, m = 20, 20

    # Create two distinct distributions
    x = keras.random.normal((n, 2), seed=42)
    y = keras.random.normal((m, 2), seed=123)

    # Get the transport plan with partial OT
    plan = sinkhorn(x, y, regularization="auto", max_steps=10_000, partial_s=s)

    if method == "log_sinkhorn":
        plan = keras.ops.exp(plan)

    # Check marginal sums: each should be approximately s/n and s/m
    row_sums = keras.ops.sum(plan, axis=1)
    col_sums = keras.ops.sum(plan, axis=0)

    expected_row_mass = s / n
    expected_col_mass = s / m

    # Each row should have approximately s/n mass (allowing small numerical error)
    assert keras.ops.all(keras.ops.abs(row_sums - expected_row_mass) < 0.05)

    # Each column should have approximately s/m mass
    assert keras.ops.all(keras.ops.abs(col_sums - expected_col_mass) < 0.05)

    # Total transported mass should be approximately s
    assert abs(float(keras.ops.sum(plan)) - s) < 1e-3
