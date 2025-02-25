import pytest


@pytest.fixture()
def batch_size():
    return 8


@pytest.fixture()
def summary_network():
    return None


@pytest.fixture()
def inference_network():
    from bayesflow.networks import CouplingFlow

    return CouplingFlow(subnet="mlp", depth=2, subnet_kwargs=dict(widths=(32, 32)))


@pytest.fixture()
def continuous_approximator(adapter, inference_network, summary_network):
    from bayesflow import ContinuousApproximator

    return ContinuousApproximator(
        adapter=adapter,
        inference_network=inference_network,
        summary_network=summary_network,
    )


@pytest.fixture()
def point_inference_network():
    from bayesflow.networks import PointInferenceNetwork
    from bayesflow.scores import NormedDifferenceScore, QuantileScore

    return PointInferenceNetwork(
        scores=dict(
            mean=NormedDifferenceScore(k=2),
            quantiles=QuantileScore(q=[0.1, 0.5, 0.9]),
        ),
        subnet="mlp",
        subnet_kwargs=dict(widths=(32, 32)),
    )


@pytest.fixture()
def point_approximator(adapter, point_inference_network, summary_network):
    from bayesflow import PointApproximator

    return PointApproximator(
        adapter=adapter,
        inference_network=point_inference_network,
        summary_network=summary_network,
    )


# @pytest.fixture(params=["continuous_approximator"], scope="function")
@pytest.fixture(params=["continuous_approximator", "point_approximator"], scope="function")
def approximator(request):
    return request.getfixturevalue(request.param)


@pytest.fixture()
def adapter():
    from bayesflow import ContinuousApproximator

    return ContinuousApproximator.build_adapter(
        inference_variables=["mean", "std"],
        inference_conditions=["x"],
    )


@pytest.fixture()
def simulator():
    from tests.utils.normal_simulator import NormalSimulator

    return NormalSimulator()


@pytest.fixture()
def train_dataset(batch_size, adapter, simulator):
    from bayesflow import OfflineDataset

    num_batches = 4
    data = simulator.sample((num_batches * batch_size,))
    return OfflineDataset(data=data, adapter=adapter, batch_size=batch_size, workers=4, max_queue_size=num_batches)


@pytest.fixture()
def validation_dataset(batch_size, adapter, simulator):
    from bayesflow import OfflineDataset

    num_batches = 2
    data = simulator.sample((num_batches * batch_size,))
    return OfflineDataset(data=data, adapter=adapter, batch_size=batch_size, workers=4, max_queue_size=num_batches)
