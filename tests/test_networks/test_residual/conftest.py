import pytest

from bayesflow.networks.helpers.residual import Residual
from bayesflow.networks.helpers import ConditionalDenseBlock


@pytest.fixture()
def residual():
    import keras

    return Residual(keras.layers.Flatten(), keras.layers.Dense(2))


@pytest.fixture()
def cond_residual():
    return ConditionalDenseBlock(width=2)


@pytest.fixture()
def build_shapes():
    return {"input_shape": (32, 2)}


@pytest.fixture()
def build_shapes_cond():
    return {"input_shape": ((32, 2), (32, 1))}
