import pytest
import numpy as np


@pytest.fixture
def simulator():
    from bayesflow import make_simulator
    from bayesflow.simulators import ModelComparisonSimulator

    def context(batch_shape, n=None):
        if n is None:
            n = np.random.randint(2, 5)
        return dict(n=n)

    def prior_null():
        return dict(mu=0.0)

    def prior_alternative():
        mu = np.random.normal(loc=0, scale=1)
        return dict(mu=mu)

    def likelihood(n, mu):
        x = np.random.normal(loc=mu, scale=1, size=n)
        return dict(x=x)

    simulator_null = make_simulator([prior_null, likelihood])
    simulator_alternative = make_simulator([prior_alternative, likelihood])
    return ModelComparisonSimulator(
        simulators=[simulator_null, simulator_alternative],
        use_mixed_batches=True,
        shared_simulator=context,
    )


@pytest.fixture
def adapter():
    from bayesflow import Adapter

    return (
        Adapter()
        .sqrt("n")
        .broadcast("n", to="x")
        .as_set("x")
        .rename("n", "classifier_conditions")
        .rename("x", "summary_variables")
        .drop("mu")
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
    from bayesflow.approximators import ModelComparisonApproximator

    return ModelComparisonApproximator(
        num_models=len(simulator.simulators),
        classifier_network=classifier_network,
        adapter=adapter,
        summary_network=summary_network,
        standardize=standardize,
    )


@pytest.fixture(
    params=["all", None, "classifier_conditions", "summary_variables", ("classifier_conditions", "summary_variables")]
)
def standardize(request):
    return request.param
