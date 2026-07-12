import pytest

from bayesflow.networks import MLP


def _make_diffusion_model(noise_schedule, prediction_type, subnet):
    """Factory for DiffusionModel instances, avoiding 6 near-identical fixtures."""
    from bayesflow.networks import DiffusionModel

    return DiffusionModel(
        subnet=subnet,
        subnet_kwargs=dict(widths=[8, 8]),
        noise_schedule=noise_schedule,
        prediction_type=prediction_type,
    )


@pytest.fixture()
def diffusion_model_edm_F():
    return _make_diffusion_model("edm", "F", "time_mlp")


@pytest.fixture()
def diffusion_model_cosine_velocity():
    return _make_diffusion_model("cosine", "velocity", "time_mlp")


@pytest.fixture()
def diffusion_model_cosine_noise():
    return _make_diffusion_model("cosine", "noise", "time_mlp")


@pytest.fixture()
def diffusion_model_edm_potential():
    return _make_diffusion_model("edm", "potential", "time_mlp")


@pytest.fixture()
def diffusion_model_transformer():
    return _make_diffusion_model("edm", "F", "diffusion_transformer")


@pytest.fixture()
def flow_matching():
    from bayesflow.networks import FlowMatching

    return FlowMatching(
        subnet_kwargs=dict(widths=[8, 8]),
    )


@pytest.fixture()
def flow_matching_transformer():
    from bayesflow.networks import FlowMatching

    return FlowMatching(
        subnet="diffusion_transformer",
        subnet_kwargs=dict(widths=[8, 8]),
    )


@pytest.fixture()
def consistency_model():
    from bayesflow.networks import ConsistencyModel

    return ConsistencyModel(
        total_steps=100,
        subnet_kwargs=dict(widths=[8, 8]),
    )


@pytest.fixture()
def stable_consistency_model():
    from bayesflow.networks import StableConsistencyModel

    return StableConsistencyModel(
        total_steps=100,
        subnet_kwargs=dict(widths=[8, 8]),
    )


@pytest.fixture()
def affine_coupling_flow():
    from bayesflow.networks import CouplingFlow

    return CouplingFlow(
        depth=2, subnet="mlp", subnet_kwargs=dict(widths=[8, 8]), transform="affine", transform_kwargs=dict(clamp=1.8)
    )


@pytest.fixture()
def spline_coupling_flow():
    from bayesflow.networks import CouplingFlow

    return CouplingFlow(
        depth=2, subnet="mlp", subnet_kwargs=dict(widths=[8, 8]), transform="spline", transform_kwargs=dict(bins=8)
    )


@pytest.fixture()
def free_form_flow():
    from bayesflow.experimental import FreeFormFlow

    return FreeFormFlow(encoder_subnet=MLP([16, 16]), decoder_subnet=MLP([16, 16]))


@pytest.fixture()
def typical_scoring_rule_network():
    from bayesflow.networks import ScoringRuleNetwork
    from bayesflow.scoring_rules import MeanScore, MedianScore, QuantileScore, MvNormalScore, MixtureScore

    return ScoringRuleNetwork(
        scoring_rules=dict(
            mean=MeanScore(),
            median=MedianScore(),
            quantiles=QuantileScore([0.1, 0.2, 0.5, 0.65]),
            mvn=MvNormalScore(),
            mix=MixtureScore(mvn_c1=MvNormalScore(), mvn_c2=MvNormalScore()),
        )
    )


@pytest.fixture()
def typical_scoring_rule_network_subnet():
    from bayesflow.networks import ScoringRuleNetwork
    from bayesflow.scoring_rules import MeanScore, MedianScore, QuantileScore, MvNormalScore, MixtureScore

    subnet = MLP([16, 8])

    return ScoringRuleNetwork(
        scoring_rules=dict(
            mean=MeanScore(subnets=dict(value=subnet)),
            median=MedianScore(subnets=dict(value=subnet)),
            quantiles=QuantileScore(subnets=dict(value=subnet)),
            mvn=MvNormalScore(subnets=dict(mean=subnet, covariance=subnet)),
            mix=MixtureScore(mvn_c1=MvNormalScore(), mvn_c2=MvNormalScore(), subnets=dict(mixture_logits=subnet)),
        ),
        subnet=subnet,
    )


@pytest.fixture()
def latent_diffusion_model():
    from bayesflow.networks import DiffusionModel, MLP, TimeMLP
    from bayesflow.experimental import LatentInferenceNetwork, VariationalAutoEncoder

    autoencoder = VariationalAutoEncoder(
        4,
        MLP([8, 8]),
        MLP([8, 8]),
    )
    inference_network = DiffusionModel(subnet=TimeMLP([8, 8]))

    return LatentInferenceNetwork(autoencoder, inference_network)


@pytest.fixture()
def latent_coupling_flow():
    from bayesflow.networks import CouplingFlow, MLP
    from bayesflow.experimental import LatentInferenceNetwork, VariationalAutoEncoder

    autoencoder = VariationalAutoEncoder(
        4,
        MLP([8, 8]),
        MLP([8, 8]),
    )
    inference_network = CouplingFlow(subnet_kwargs=dict(widths=[8, 8]))

    return LatentInferenceNetwork(autoencoder, inference_network)


@pytest.fixture(
    params=[
        "typical_scoring_rule_network",
        "affine_coupling_flow",
        "spline_coupling_flow",
        "flow_matching",
        "free_form_flow",
        "consistency_model",
        "stable_consistency_model",
        "latent_diffusion_model",
        "latent_coupling_flow",
        pytest.param("diffusion_model_edm_F"),
        pytest.param("diffusion_model_transformer"),
        pytest.param("diffusion_model_cosine_velocity", marks=pytest.mark.slow),
        pytest.param("diffusion_model_cosine_noise", marks=pytest.mark.slow),
        pytest.param("diffusion_model_edm_potential", marks=pytest.mark.slow),
    ],
    scope="function",
)
def inference_network(request):
    return request.getfixturevalue(request.param)


@pytest.fixture(
    params=[
        "affine_coupling_flow",
        "spline_coupling_flow",
        "flow_matching",
        "free_form_flow",
        "consistency_model",
        "stable_consistency_model",
        pytest.param("diffusion_model_edm_F"),
        pytest.param("diffusion_model_transformer"),
        pytest.param("flow_matching_transformer"),
        pytest.param("diffusion_model_cosine_velocity", marks=pytest.mark.slow),
        pytest.param("diffusion_model_edm_potential", marks=pytest.mark.slow),
    ],
    scope="function",
)
def generative_inference_network(request):
    return request.getfixturevalue(request.param)


@pytest.fixture(
    params=[
        "flow_matching",
        "consistency_model",
        "stable_consistency_model",
        "diffusion_model",
        "flow_matching_transformer",
        "consistency_model_transformer",
        "stable_consistency_model_transformer",
        "diffusion_model_transformer",
    ],
    scope="function",
)
def diffusion_type_inference_network(request):
    if request.param == "flow_matching":
        from bayesflow.networks import FlowMatching

        network = FlowMatching
    elif request.param == "consistency_model":
        from bayesflow.networks import ConsistencyModel

        network = ConsistencyModel
    elif request.param == "stable_consistency_model":
        from bayesflow.networks import StableConsistencyModel

        network = StableConsistencyModel
    elif request.param == "diffusion_model":
        from bayesflow.networks import DiffusionModel

        network = DiffusionModel
    elif request.param == "flow_matching_transformer":
        import functools

        from bayesflow.networks import FlowMatching

        network = functools.partial(FlowMatching, subnet="diffusion_transformer")
    elif request.param == "consistency_model_transformer":
        import functools

        from bayesflow.networks import ConsistencyModel

        network = functools.partial(ConsistencyModel, subnet="diffusion_transformer")
    elif request.param == "stable_consistency_model_transformer":
        import functools

        from bayesflow.networks import StableConsistencyModel

        network = functools.partial(StableConsistencyModel, subnet="diffusion_transformer")
    elif request.param == "diffusion_model_transformer":
        import functools

        from bayesflow.networks import DiffusionModel

        network = functools.partial(DiffusionModel, subnet="diffusion_transformer")
    else:
        raise ValueError(f"Unknown request param: {request.param}")
    return network


@pytest.fixture(scope="function")
def time_series_network(summary_dim):
    from bayesflow.networks import TimeSeriesNetwork

    return TimeSeriesNetwork(summary_dim=summary_dim)


@pytest.fixture(scope="function")
def time_series_transformer(summary_dim):
    from bayesflow.networks import TimeSeriesTransformer

    # return_sequences=False to act as a regular summary (compression) network
    return TimeSeriesTransformer(summary_dim=summary_dim, return_sequences=False)


@pytest.fixture(scope="function")
def fusion_transformer(summary_dim):
    from bayesflow.networks import FusionTransformer

    return FusionTransformer(summary_dim=summary_dim)


@pytest.fixture(scope="function")
def set_transformer(summary_dim):
    from bayesflow.networks import SetTransformer

    return SetTransformer(summary_dim=summary_dim)


@pytest.fixture(scope="function")
def deep_set(summary_dim):
    from bayesflow.networks import DeepSet

    return DeepSet(summary_dim=summary_dim)


@pytest.fixture(
    params=[
        None,
        "time_series_network",
        "time_series_transformer",
        "fusion_transformer",
        "set_transformer",
        "deep_set",
    ],
    scope="function",
)
def summary_network(request, summary_dim):
    if request.param is None:
        return None
    return request.getfixturevalue(request.param)
