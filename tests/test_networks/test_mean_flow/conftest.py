import keras
import pytest


# ---------------------------------------------------------------------------
# Shapes
# ---------------------------------------------------------------------------


@pytest.fixture(params=[2], scope="session")
def batch_size(request):
    return request.param


@pytest.fixture(params=[2], scope="session")
def xz_dim(request):
    return request.param


@pytest.fixture(params=[None, 3], scope="session")
def cond_dim(request):
    return request.param


@pytest.fixture(scope="session")
def random_samples(batch_size, xz_dim):
    return keras.random.normal((batch_size, xz_dim))


@pytest.fixture(scope="session")
def random_conditions(batch_size, cond_dim):
    if cond_dim is None:
        return None
    return keras.random.normal((batch_size, cond_dim))


# ---------------------------------------------------------------------------
# MeanFlow variants
# ---------------------------------------------------------------------------


@pytest.fixture(
    params=[
        dict(),
        dict(velocity_head_layers=0),
        dict(cycle_weight=0.0),
        dict(use_optimal_transport=True),
        dict(fixed_target_prob=0.3, missing_target_prob=0.3, missing_conditions_prob=0.3),
        dict(loss_fn="mse", velocity_loss_fn="adaptive_l2"),
        dict(density_head=True, fixed_target_prob=0.3),
    ],
    ids=["default", "no_head", "no_cycle", "ot", "masking", "losses", "density_head"],
)
def mean_flow(request):
    from bayesflow.experimental import MeanFlow

    return MeanFlow(subnet_kwargs=dict(widths=(8, 8)), **request.param)
