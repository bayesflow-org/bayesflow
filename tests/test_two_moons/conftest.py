import pytest


@pytest.fixture()
def inference_network():
    from bayesflow.networks import CouplingFlow

    return CouplingFlow(depth=2, subnet="mlp", subnet_kwargs=dict(widths=(32, 32)))


@pytest.fixture()
def approximator(adapter, inference_network):
    from bayesflow import ContinuousApproximator

    return ContinuousApproximator(
        adapter=adapter,
        inference_network=inference_network,
    )


@pytest.fixture()
def batch_size():
    return 128


@pytest.fixture()
def adapter():
    from bayesflow import ContinuousApproximator

    return ContinuousApproximator.build_adapter(
        inference_variables=["parameters"],
        inference_conditions=["observables"],
    )


@pytest.fixture()
def random_samples(batch_size, simulator):
    return simulator.sample((batch_size,))


@pytest.fixture()
def simulator():
    from bayesflow.simulators import TwoMoons

    return TwoMoons()


@pytest.fixture()
def train_dataset(batch_size, adapter, simulator):
    from bayesflow import OfflineDataset

    num_batches = 32
    data = simulator.sample((num_batches * batch_size,))
    return OfflineDataset(
        data=data,
        adapter=adapter,
        batch_size=batch_size,
        workers=4,
        max_queue_size=num_batches,
        use_multiprocessing=False,
    )


@pytest.fixture()
def validation_dataset(batch_size, adapter, simulator):
    from bayesflow import OfflineDataset

    num_batches = 8
    data = simulator.sample((num_batches * batch_size,))
    return OfflineDataset(data=data, adapter=adapter, batch_size=batch_size, workers=4, max_queue_size=num_batches)
