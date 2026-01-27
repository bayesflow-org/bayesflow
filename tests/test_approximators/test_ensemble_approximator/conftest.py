import pytest
from tests.utils import check_combination_simulator_adapter


@pytest.fixture()
def train_dataset_for_ensemble(batch_size, adapter, simulator):
    check_combination_simulator_adapter(simulator, adapter)

    from bayesflow import OfflineDataset, EnsembleDataset

    num_batches = 4
    data = simulator.sample((num_batches * batch_size,))
    return EnsembleDataset(
        OfflineDataset(data=data, adapter=adapter, batch_size=batch_size, workers=4, max_queue_size=num_batches),
        num_ensemble=2,
    )


@pytest.fixture()
def ensemble_approximator_continuous_and_point(continuous_approximator, point_approximator_without_parametric_score):
    from bayesflow import EnsembleApproximator

    return EnsembleApproximator(
        dict(cont_approx=continuous_approximator, point_approx=point_approximator_without_parametric_score)
    )


@pytest.fixture(
    params=[
        "ensemble_approximator_continuous_and_point",
    ],
    scope="function",
)
def ensemble_approximator(request):
    return request.getfixturevalue(request.param)
