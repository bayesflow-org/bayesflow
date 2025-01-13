import pytest


@pytest.fixture()
def actnorm():
    from bayesflow.networks.coupling_flow.actnorm import ActNorm

    return ActNorm()


@pytest.fixture()
def dual_coupling():
    from bayesflow.networks.coupling_flow.couplings import DualCoupling

    return DualCoupling(transform="spline")


@pytest.fixture(params=["actnorm", "dual_coupling"])
def invertible_layer(request):
    return request.getfixturevalue(request.param)


@pytest.fixture()
def single_coupling():
    from bayesflow.networks.coupling_flow.couplings import SingleCoupling

    return SingleCoupling()
