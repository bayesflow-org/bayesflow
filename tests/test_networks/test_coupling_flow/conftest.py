import pytest


@pytest.fixture()
def actnorm():
    from bayesflow.networks.coupling_flow.actnorm import ActNorm

    return ActNorm()


@pytest.fixture()
def dual_coupling(request, transform):
    from bayesflow.networks.coupling_flow.couplings import DualCoupling

    return DualCoupling(transform=transform)


@pytest.fixture(params=["actnorm", "dual_coupling"])
def invertible_layer(request, transform):
    return request.getfixturevalue(request.param)


@pytest.fixture()
def single_coupling(request, transform):
    from bayesflow.networks.coupling_flow.couplings import SingleCoupling

    return SingleCoupling(transform=transform)


@pytest.fixture(params=["affine", "spline"])
def transform(request):
    return request.param
