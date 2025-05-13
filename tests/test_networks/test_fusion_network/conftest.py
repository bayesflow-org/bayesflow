import pytest


@pytest.fixture()
def multimodal_data(random_samples, random_set):
    return {"x1": random_samples, "x2": random_set}


@pytest.fixture()
def fusion_network():
    from bayesflow.networks import FusionNetwork, DeepSet
    import keras

    return FusionNetwork(
        backbones={"x1": keras.layers.Dense(3), "x2": DeepSet(summary_dim=2, base_distribution="normal")},
        head=keras.layers.Dense(3),
    )
