import pytest

from bayesflow.networks import Sequential


@pytest.fixture()
def sequential():
    import keras

    return Sequential(keras.layers.Flatten(), keras.layers.Dense(2))


@pytest.fixture()
def build_shapes():
    return {"input_shape": (32, 2)}
