import pytest


# For the serialization tests, we want to test passing str and type.
# For all other tests, this is not necessary and would double test time.
# Therefore, below we specify two variants of each network, one without and
# one with a subnet parameter. The latter will only be used for the relevant
# tests. If there is a better way to set the params to a single value ("mlp")
# for a given test, maybe this can be simplified, but I did not see one.
@pytest.fixture(params=["str", "type"], scope="function")
def subnet(request):
    if request.param == "str":
        return "mlp"

    from bayesflow.networks import MLP

    return MLP


@pytest.fixture()
def flow_matching():
    from bayesflow.networks import FlowMatching

    return FlowMatching(
        subnet_kwargs={"widths": None, "width": 64, "depth": 2},
        integrate_kwargs={"method": "rk45", "steps": 100},
    )


@pytest.fixture()
def flow_matching_subnet(subnet):
    from bayesflow.networks import FlowMatching

    return FlowMatching(subnet=subnet)


@pytest.fixture()
def coupling_flow():
    from bayesflow.networks import CouplingFlow

    return CouplingFlow()


@pytest.fixture()
def coupling_flow_subnet(subnet):
    from bayesflow.networks import CouplingFlow

    return CouplingFlow(subnet=subnet)


@pytest.fixture()
def free_form_flow():
    from bayesflow.experimental import FreeFormFlow

    return FreeFormFlow()


@pytest.fixture()
def free_form_flow_subnet(subnet):
    from bayesflow.experimental import FreeFormFlow

    return FreeFormFlow(encoder_subnet=subnet, decoder_subnet=subnet)


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
def typical_point_inference_network_subnet(subnet):
    from bayesflow.networks import PointInferenceNetwork
    from bayesflow.scores import MeanScore, MedianScore, QuantileScore, MultivariateNormalScore

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
def lst_net(summary_dim):
    from bayesflow.networks import LSTNet

    return LSTNet(summary_dim=summary_dim)


@pytest.fixture(scope="function")
def set_transformer(summary_dim):
    from bayesflow.networks import SetTransformer

    return SetTransformer(summary_dim=summary_dim)


@pytest.fixture(scope="function")
def deep_set(summary_dim):
    from bayesflow.networks import DeepSet

    return DeepSet(summary_dim=summary_dim)


@pytest.fixture(params=[None, "lst_net", "set_transformer", "deep_set"], scope="function")
def summary_network(request, summary_dim):
    if request.param is None:
        return None
    return request.getfixturevalue(request.param)
