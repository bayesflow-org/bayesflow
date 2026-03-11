import keras
import pytest


# ---------------------------------------------------------------------------
# Shapes
# ---------------------------------------------------------------------------


@pytest.fixture(params=[2], scope="session")
def batch_size(request):
    return request.param


@pytest.fixture(params=[2, 5], scope="session")
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
# FlowMatching variants
# ---------------------------------------------------------------------------


@pytest.fixture(
    params=[
        dict(),
        dict(use_optimal_transport=True),
        dict(loss_fn="mae"),
        dict(time_power_law_alpha=1.0),
    ],
    ids=["default", "ot", "mae_loss", "power_law"],
)
def flow_matching(request):
    from bayesflow.networks import FlowMatching

    return FlowMatching(subnet_kwargs=dict(widths=(8, 8)), **request.param)


@pytest.fixture()
def flow_matching_with_masking():
    from bayesflow.networks import FlowMatching

    return FlowMatching(
        subnet_kwargs=dict(widths=(8, 8)),
        drop_cond_prob=0.1,
        drop_target_prob=0.5,
    )
