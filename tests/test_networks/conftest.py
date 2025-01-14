import pytest


@pytest.fixture()
def deep_set():
    from bayesflow.networks import DeepSet

    return DeepSet()


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

    return FlowMatching()


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
    from bayesflow.networks import FreeFormFlow

    return FreeFormFlow()


@pytest.fixture()
def free_form_flow_subnet(subnet):
    from bayesflow.networks import FreeFormFlow

    return FreeFormFlow(encoder_subnet=subnet, decoder_subnet=subnet)


@pytest.fixture(params=["coupling_flow", "flow_matching", "free_form_flow"], scope="function")
def inference_network(request):
    return request.getfixturevalue(request.param)


@pytest.fixture(params=["coupling_flow_subnet", "flow_matching_subnet", "free_form_flow_subnet"], scope="function")
def inference_network_subnet(request):
    return request.getfixturevalue(request.param)


@pytest.fixture()
def lst_net():
    from bayesflow.networks import LSTNet

    return LSTNet()


@pytest.fixture()
def set_transformer():
    from bayesflow.networks import SetTransformer

    return SetTransformer()


@pytest.fixture(params=[None, "deep_set", "lst_net", "set_transformer"])
def summary_network(request):
    if request.param is None:
        return None

    return request.getfixturevalue(request.param)
