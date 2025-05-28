import pytest

from bayesflow.networks import MLP


@pytest.fixture(params=[None, 0.0, 0.1])
def dropout(request):
    return request.param


@pytest.fixture(params=[None, "batch"])
def norm(request):
    return request.param


@pytest.fixture(params=[False, True])
def residual(request):
    return request.param


@pytest.fixture()
def mlp(dropout, norm, residual):
    return MLP([64, 64], dropout=dropout, norm=norm, residual=residual)


@pytest.fixture()
def build_shapes():
    return {"input_shape": (32, 2)}
