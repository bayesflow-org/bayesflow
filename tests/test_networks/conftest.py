import pytest

from bayesflow.networks import MLP


def _make_diffusion_model(noise_schedule, prediction_type):
    """Factory for DiffusionModel instances, avoiding 6 near-identical fixtures."""
    from bayesflow.networks import DiffusionModel

    return DiffusionModel(
        subnet_kwargs=dict(widths=[8, 8]),
        noise_schedule=noise_schedule,
        prediction_type=prediction_type,
    )


@pytest.fixture()
def diffusion_model_edm_F():
    return _make_diffusion_model("edm", "F")


@pytest.fixture()
def diffusion_model_edm_velocity():
    return _make_diffusion_model("edm", "velocity")


@pytest.fixture()
def diffusion_model_cosine_velocity():
    return _make_diffusion_model("cosine", "velocity")


@pytest.fixture()
def diffusion_model_cosine_noise():
    return _make_diffusion_model("cosine", "noise")


@pytest.fixture()
def flow_matching():
    from bayesflow.networks import FlowMatching

    return FlowMatching(
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


@pytest.fixture(
    params=[
        "typical_scoring_rule_network",
        "affine_coupling_flow",
        "spline_coupling_flow",
        "flow_matching",
        "free_form_flow",
        "consistency_model",
        "stable_consistency_model",
        pytest.param("diffusion_model_edm_F"),
        pytest.param("diffusion_model_cosine_velocity", marks=pytest.mark.slow),
        pytest.param("diffusion_model_cosine_noise", marks=pytest.mark.slow),
    ],
    scope="function",
)
def inference_network(request):
    return request.getfixturevalue(request.param)


@pytest.fixture(
    params=[
        "typical_scoring_rule_network_subnet",
        "coupling_flow_subnet",
        "flow_matching_subnet",
        "free_form_flow_subnet",
    ],
    scope="function",
)
def inference_network_subnet(request):
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
        pytest.param("diffusion_model_cosine_velocity", marks=pytest.mark.slow),
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
