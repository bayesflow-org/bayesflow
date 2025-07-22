import pytest
import numpy as np
from tests.utils import check_combination_simulator_adapter


@pytest.fixture()
def train_dataset_for_ensemble(batch_size, adapter, simulator):
    check_combination_simulator_adapter(simulator, adapter)

    from bayesflow import OfflineEnsembleDataset

    num_batches = 4
    data = simulator.sample((num_batches * batch_size,))
    return OfflineEnsembleDataset(
        num_ensemble=2, data=data, adapter=adapter, batch_size=batch_size, workers=4, max_queue_size=num_batches
    )


@pytest.fixture()
def continuous_and_point_approximator_ensemble(
    continuous_approximator, point_approximator_with_single_parametric_score
):
    from bayesflow import ApproximatorEnsemble

    return ApproximatorEnsemble(
        dict(cont_approx=continuous_approximator, point_approx=point_approximator_with_single_parametric_score)
    )


@pytest.fixture(
    params=[
        "continuous_and_point_approximator_ensemble",
    ],
    scope="function",
)
def continuous_approximator_ensemble(request):
    return request.getfixturevalue(request.param)


@pytest.fixture
def model_comparison_simulator():
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


@pytest.fixture()
def model_comparison_train_dataset_for_ensemble(batch_size, adapter, simulator):
    check_combination_simulator_adapter(simulator, adapter)

    from bayesflow import OfflineEnsembleDataset

    num_batches = 4
    data = simulator.sample((num_batches * batch_size,))
    return OfflineEnsembleDataset(
        num_ensemble=2, data=data, adapter=adapter, batch_size=batch_size, workers=4, max_queue_size=num_batches
    )


@pytest.fixture
def model_comparison_adapter():
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


@pytest.fixture()
def basic_model_comparison_ensemble(model_comparison_adapter):
    from bayesflow.approximators import ModelComparisonApproximator, ApproximatorEnsemble
    from bayesflow.networks import DeepSet, MLP

    classifier_network = MLP(widths=[32, 32])

    summary_network = DeepSet(summary_dim=2, depth=1)

    approx_1 = ModelComparisonApproximator(
        num_models=2,
        classifier_network=classifier_network,
        adapter=model_comparison_adapter,
        summary_network=summary_network,
    )
    approx_2 = ModelComparisonApproximator(
        num_models=2,
        classifier_network=classifier_network,
        adapter=model_comparison_adapter,
        summary_network=summary_network,
    )

    return ApproximatorEnsemble(dict(approx_1=approx_1, approx_2=approx_2))


@pytest.fixture(
    params=[
        "basic_model_comparison_ensemble",
    ],
    scope="function",
)
def model_comparison_approximator_ensemble(request):
    return request.getfixturevalue(request.param)


@pytest.fixture(
    params=[
        "continuous_and_point_approximator_ensemble",
        "basic_model_comparison_ensemble",
    ],
    scope="function",
)
def approximator_ensemble(request):
    return request.getfixturevalue(request.param)
