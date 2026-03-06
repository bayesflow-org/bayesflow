import pytest
import numpy as np


@pytest.fixture
def prior():
    def fn():
        mu = np.random.normal(loc=0, scale=1)
        return dict(mu=mu)

    return fn


@pytest.fixture
def likelihood():
    def fn(mu):
        x = np.random.normal(loc=mu, scale=0.25)
        return dict(x=x)

    return fn


@pytest.fixture
def simulator(prior, likelihood):
    from bayesflow import make_simulator

    return make_simulator([prior, likelihood])


@pytest.fixture
def adapter():
    from bayesflow import Adapter

    return (
        Adapter()
        .convert_dtype("float64", "float32")
        .rename("x", "inference_conditions")
        .rename("mu", "inference_variables")
    )


@pytest.fixture
def classifier_network():
    from bayesflow.networks import MLP

    return MLP(widths=(8, 8))


@pytest.fixture
def approximator(adapter, classifier_network, standardize):
    from bayesflow.approximators import RatioApproximator

    return RatioApproximator(
        inference_network=classifier_network,
        adapter=adapter,
        standardize=standardize,
    )


@pytest.fixture(
    params=["all", None, "inference_conditions", "inference_variables", ("inference_conditions", "inference_variables")]
)
def standardize(request):
    return request.param
