import pytest

from bayesflow.networks.residual import Residual


@pytest.fixture()
def residual():
    import keras

    return Residual(keras.layers.Flatten(), keras.layers.Dense(2))


@pytest.fixture()
def build_shapes():
    return {"input_shape": (32, 2)}
