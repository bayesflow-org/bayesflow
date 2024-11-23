import pytest


@pytest.fixture()
def deep_set():
    from bayesflow.networks import DeepSet

    return DeepSet()


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
