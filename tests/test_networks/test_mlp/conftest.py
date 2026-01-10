import pytest

from bayesflow.networks import MLP, TimeMLP


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
def time_mlp(dropout, norm, residual):
    return TimeMLP(widths=[64, 64], dropout=dropout, norm=norm, residual=residual)


@pytest.fixture()
def build_shapes():
    return {"input_shape": (32, 2)}


@pytest.fixture()
def build_shapes_time():
    return {"x_shape": (32, 2), "t_shape": (32, 1), "conditions_shape": (32, 4)}
