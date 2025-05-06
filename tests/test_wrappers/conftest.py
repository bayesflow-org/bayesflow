import pytest


@pytest.fixture()
def inference_network():
    from bayesflow.networks import CouplingFlow

    return CouplingFlow(depth=2)


@pytest.fixture()
def random_time_series():
    import keras

    return keras.random.normal(shape=(2, 80, 2))


@pytest.fixture()
def mamba_summary_network():
    from bayesflow.wrappers.mamba import Mamba

    return Mamba(summary_dim=4, feature_dims=(2, 2), state_dims=(4, 4), conv_dims=(8, 8))
