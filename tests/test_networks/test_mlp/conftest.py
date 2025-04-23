import pytest

from bayesflow.networks import MLP


@pytest.fixture()
def mlp():
    return MLP([64, 64])


@pytest.fixture()
def build_shapes():
    return {"input_shape": (32, 2)}
