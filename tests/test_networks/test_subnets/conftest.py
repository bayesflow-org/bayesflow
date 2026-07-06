import pytest

from bayesflow.networks import MLP, TimeMLP, DiffusionTransformer


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
def diffusion_transformer(dropout):
    return DiffusionTransformer(widths=[64, 64], num_heads=4, dropout=0.0 if dropout is None else dropout)


@pytest.fixture()
def build_shapes():
    return {"input_shape": (32, 2)}


@pytest.fixture()
def build_shapes_time():
    return {"input_shape": ((32, 2), (32, 1), (32, 4))}
