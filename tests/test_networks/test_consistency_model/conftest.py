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
# ConsistencyModel variants
# ---------------------------------------------------------------------------


@pytest.fixture(
    params=[
        dict(total_steps=100),
        dict(total_steps=100, max_time=40),
        dict(total_steps=200, s0=5, s1=50),
    ],
    ids=["default", "max_time_40", "custom_schedule"],
)
def consistency_model(request):
    from bayesflow.networks import ConsistencyModel

    return ConsistencyModel(subnet_kwargs=dict(widths=(8, 8)), **request.param)


@pytest.fixture()
def consistency_model_with_masking():
    from bayesflow.networks import ConsistencyModel

    return ConsistencyModel(total_steps=100, subnet_kwargs=dict(widths=(8, 8)), drop_cond_prob=0.1)


# ---------------------------------------------------------------------------
# StableConsistencyModel variants
# ---------------------------------------------------------------------------


@pytest.fixture(
    params=[
        dict(),
        dict(sigma=2.0),
    ],
    ids=["default", "sigma_2"],
)
def stable_consistency_model(request):
    from bayesflow.networks import StableConsistencyModel

    return StableConsistencyModel(subnet_kwargs=dict(widths=(8, 8)), **request.param)


@pytest.fixture()
def stable_consistency_model_with_masking():
    from bayesflow.networks import StableConsistencyModel

    return StableConsistencyModel(subnet_kwargs=dict(widths=(8, 8)), drop_cond_prob=0.1)
