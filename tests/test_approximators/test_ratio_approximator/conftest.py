import pytest
import numpy as np


@pytest.fixture
def context():
    def fn():
        n = np.random.randint(2, 5)
        return dict(n=n)

    return fn


@pytest.fixture
def prior():
    def fn():
        mu = np.random.normal(loc=0, scale=1)
        return dict(mu=mu)

    return fn


@pytest.fixture
def likelihood():
    def fn(n, mu):
        x = np.random.normal(loc=mu, scale=0.25, size=n)
        return dict(x=x)

    return fn


@pytest.fixture
def simulator(context, prior, likelihood):
    from bayesflow import make_simulator

    return make_simulator([prior, likelihood], meta_fn=context)


@pytest.fixture
def adapter():
    from bayesflow import Adapter

    return (
        Adapter()
        .drop("n")
        .as_set("x")
        .rename("x", "inference_conditions")
        .rename("mu", "inference_variables")
        .convert_dtype("float64", "float32")
    )


@pytest.fixture
def summary_network():
    from bayesflow.networks import DeepSet

    return DeepSet(summary_dim=2, depth=1)


@pytest.fixture
def classifier_network():
    from bayesflow.networks import MLP

    return MLP(widths=[32, 32])


@pytest.fixture
def approximator(adapter, classifier_network, summary_network, simulator, standardize):
    from bayesflow.approximators import RatioApproximator

    return RatioApproximator(
        classifier_network=classifier_network,
        adapter=adapter,
        summary_network=summary_network,
        standardize=standardize,
    )


@pytest.fixture(
    params=["all", None, "inference_conditions", "inference_variables", ("inference_conditions", "inference_variables")]
)
def standardize(request):
    return request.param
