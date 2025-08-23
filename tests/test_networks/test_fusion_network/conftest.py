import pytest


@pytest.fixture(params=[True, False])
def multimodal(request):
    return request.param


@pytest.fixture()
def data(random_samples, random_set, multimodal):
    if multimodal:
        return {"x1": random_samples, "x2": random_set}
    return random_set


@pytest.fixture()
def fusion_network(multimodal):
    from bayesflow.networks import FusionNetwork, DeepSet
    import keras

    deepset_kwargs = dict(
        summary_dim=2,
        mlp_widths_equivariant=(2, 2),
        mlp_widths_invariant_inner=(2, 2),
        mlp_widths_invariant_outer=(2, 2),
        mlp_widths_invariant_last=(2, 2),
        base_distribution="normal",
    )
    if multimodal:
        return FusionNetwork(
            backbones={"x1": keras.layers.Dense(3), "x2": DeepSet(**deepset_kwargs)},
            head=keras.layers.Dense(3),
        )
    return FusionNetwork(
        backbones=[
            DeepSet(**deepset_kwargs),
            DeepSet(**deepset_kwargs),
        ],
        head=keras.layers.Dense(3),
    )
