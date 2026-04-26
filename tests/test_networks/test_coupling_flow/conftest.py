import keras
import pytest


@pytest.fixture(params=[2], scope="session")
def batch_size(request):
    return request.param


@pytest.fixture(params=[4, 5], scope="session")
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


@pytest.fixture(params=["affine", "spline"])
def transform(request):
    return request.param


@pytest.fixture()
def actnorm():
    from bayesflow.networks.inference.coupling.actnorm import ActNorm

    return ActNorm()


@pytest.fixture()
def dual_coupling(request, transform):
    from bayesflow.networks.inference.coupling.layers import DualCoupling

    return DualCoupling(transform=transform)


@pytest.fixture(params=["actnorm", "dual_coupling"])
def invertible_layer(request, transform):
    return request.getfixturevalue(request.param)


@pytest.fixture()
def single_coupling(request, transform):
    from bayesflow.networks.inference.coupling.layers import SingleCoupling

    return SingleCoupling(transform=transform)


@pytest.fixture(
    params=[
        dict(transform="affine"),
        dict(transform="spline", transform_kwargs=dict(bins=8)),
    ],
    ids=["affine", "spline"],
)
def coupling_flow(request):
    from bayesflow.networks import CouplingFlow

    return CouplingFlow(
        depth=2,
        subnet="mlp",
        subnet_kwargs=dict(widths=(8, 8)),
        **request.param,
    )
