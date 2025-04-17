import pytest

from bayesflow.networks import MLP


@pytest.fixture()
def flow_matching():
    from bayesflow.networks import FlowMatching

    return FlowMatching(
        subnet=MLP([64, 64]),
        integrate_kwargs={"method": "rk45", "steps": 100},
    )


@pytest.fixture()
def coupling_flow():
    from bayesflow.networks import CouplingFlow

    return CouplingFlow(depth=2, subnet="mlp", subnet_kwargs=dict(widths=[64, 64]))


@pytest.fixture()
def free_form_flow():
    from bayesflow.experimental import FreeFormFlow

    return FreeFormFlow(encoder_subnet=MLP([64, 64]), decoder_subnet=MLP([64, 64]))


@pytest.fixture()
def typical_point_inference_network():
    from bayesflow.networks import PointInferenceNetwork
    from bayesflow.scores import MeanScore, MedianScore, QuantileScore, MultivariateNormalScore

    return PointInferenceNetwork(
        scores=dict(
            mean=MeanScore(),
            median=MedianScore(),
            quantiles=QuantileScore([0.1, 0.2, 0.5, 0.65]),
            mvn=MultivariateNormalScore(),  # currently not stable
        )
    )


@pytest.fixture()
def typical_point_inference_network_subnet():
    from bayesflow.networks import PointInferenceNetwork
    from bayesflow.scores import MeanScore, MedianScore, QuantileScore, MultivariateNormalScore

    subnet = MLP([64, 64])

    return PointInferenceNetwork(
        scores=dict(
            mean=MeanScore(subnets=dict(value=subnet)),
            median=MedianScore(subnets=dict(value=subnet)),
            quantiles=QuantileScore(subnets=dict(value=subnet)),
            mvn=MultivariateNormalScore(subnets=dict(mean=subnet, covariance=subnet)),
        ),
        subnet=subnet,
    )


@pytest.fixture(
    params=["typical_point_inference_network", "coupling_flow", "flow_matching", "free_form_flow"], scope="function"
)
def inference_network(request):
    return request.getfixturevalue(request.param)


@pytest.fixture(
    params=[
        "typical_point_inference_network_subnet",
        "coupling_flow_subnet",
        "flow_matching_subnet",
        "free_form_flow_subnet",
    ],
    scope="function",
)
def inference_network_subnet(request):
    return request.getfixturevalue(request.param)


@pytest.fixture(params=["coupling_flow", "flow_matching", "free_form_flow"], scope="function")
def generative_inference_network(request):
    return request.getfixturevalue(request.param)


@pytest.fixture(scope="function")
def time_series_network(summary_dim):
    from bayesflow.networks import TimeSeriesNetwork

    return TimeSeriesNetwork(summary_dim=summary_dim)


@pytest.fixture(scope="function")
def set_transformer(summary_dim):
    from bayesflow.networks import SetTransformer

    return SetTransformer(summary_dim=summary_dim)


@pytest.fixture(scope="function")
def deep_set(summary_dim):
    from bayesflow.networks import DeepSet

    return DeepSet(summary_dim=summary_dim)


@pytest.fixture(params=[None, "time_series_network", "set_transformer", "deep_set"], scope="function")
def summary_network(request, summary_dim):
    if request.param is None:
        return None
    return request.getfixturevalue(request.param)
