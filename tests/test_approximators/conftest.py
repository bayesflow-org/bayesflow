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
def approximator(data_adapter, inference_network, summary_network):
    from bayesflow import ContinuousApproximator

    return ContinuousApproximator(
        data_adapter=data_adapter,
        inference_network=inference_network,
        summary_network=summary_network,
    )


@pytest.fixture()
def data_adapter():
    from bayesflow import ContinuousApproximator

    return ContinuousApproximator.build_data_adapter(
        inference_variables=["mean", "std"],
        inference_conditions=["x"],
    )


@pytest.fixture()
def simulator():
    from tests.utils.normal_simulator import NormalSimulator

    return NormalSimulator()


@pytest.fixture()
def train_dataset(batch_size, data_adapter, simulator):
    from bayesflow import OfflineDataset

    num_batches = 4
    data = simulator.sample((num_batches * batch_size,))
    return OfflineDataset(
        data=data, data_adapter=data_adapter, batch_size=batch_size, workers=4, max_queue_size=num_batches
    )


@pytest.fixture()
def validation_dataset(batch_size, data_adapter, simulator):
    from bayesflow import OfflineDataset

    num_batches = 2
    data = simulator.sample((num_batches * batch_size,))
    return OfflineDataset(
        data=data, data_adapter=data_adapter, batch_size=batch_size, workers=4, max_queue_size=num_batches
    )
