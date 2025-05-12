import pytest


@pytest.fixture()
def multimodal_data(random_samples, random_set):
    return {"x1": random_samples, "x2": random_set}


@pytest.fixture()
def multimodal_summary_network():
    from bayesflow.networks import MultimodalSummaryNetwork, DeepSet
    import keras

    return MultimodalSummaryNetwork(
        summary_networks={"x1": keras.layers.Dense(3), "x2": DeepSet(summary_dim=2, base_distribution="normal")},
        fusion_network=keras.layers.Dense(3),
    )
