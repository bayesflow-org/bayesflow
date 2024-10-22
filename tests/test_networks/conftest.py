import pytest


@pytest.fixture(params=[1, 2, 16], scope="session")
def summary_dim(request):
    return request.param


@pytest.fixture(params=[1, 2, 16], scope="session")
def key_dim(request):
    return request.param


@pytest.fixture(scope="function")
def lst_net(summary_dim):
    from bayesflow.networks import LSTNet

    return LSTNet(summary_dim=summary_dim)


@pytest.fixture(scope="function")
def set_transformer(summary_dim):
    from bayesflow.networks import SetTransformer

    return SetTransformer(summary_dim=summary_dim)


@pytest.fixture(scope="function")
def set_transformer_key_dim_variation(summary_dim, key_dim):
    from bayesflow.networks import SetTransformer

    return SetTransformer(summary_dim=summary_dim, key_dim=key_dim)


@pytest.fixture(scope="function")
def deep_set(summary_dim):
    from bayesflow.networks import DeepSet

    return DeepSet(summary_dim=summary_dim)


@pytest.fixture(params=[None, "lst_net", "set_transformer", "deep_set"], scope="function")
def summary_network(request, summary_dim):
    if request.param is None:
        return None
    return request.getfixturevalue(request.param)
