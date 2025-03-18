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
def approximator(adapter, inference_network, summary_network):
    from bayesflow import ContinuousApproximator

    return ContinuousApproximator(
        adapter=adapter,
        inference_network=inference_network,
        summary_network=summary_network,
    )


@pytest.fixture()
def adapter_without_sample_weight():
    from bayesflow import ContinuousApproximator

    return ContinuousApproximator.build_adapter(
        inference_variables=["mean", "std"],
        inference_conditions=["x"],
    )


@pytest.fixture()
def adapter_with_sample_weight():
    from bayesflow import ContinuousApproximator

    return ContinuousApproximator.build_adapter(
        inference_variables=["mean", "std"],
        inference_conditions=["x"],
        sample_weight="weight",
    )


@pytest.fixture(params=["adapter_without_sample_weight", "adapter_with_sample_weight"])
def adapter(request):
    return request.getfixturevalue(request.param)


@pytest.fixture()
def normal_simulator():
    from tests.utils.normal_simulator import NormalSimulator

    return NormalSimulator()


@pytest.fixture()
def normal_simulator_with_sample_weight():
    from tests.utils.normal_simulator import NormalSimulator
    from bayesflow import make_simulator

    def weight(mean):
        return dict(weight=1.0)

    return make_simulator([NormalSimulator(), weight])


@pytest.fixture(params=["normal_simulator", "normal_simulator_with_sample_weight"])
def simulator(request):
    return request.getfixturevalue(request.param)


@pytest.fixture()
def train_dataset(batch_size, adapter, simulator):
    # scan adapter representation for occurance of a rename pattern for 'sample_weight'
    adapter_with_sample_weight = "-> 'sample_weight'" in str(adapter)
    # check whether the simulator returns a 'weight' key
    simulator_with_sample_weight = "weight" in simulator.sample(1).keys()

    if adapter_with_sample_weight and not simulator_with_sample_weight:
        # adapter should expect a 'weight' key and raise a KeyError.
        with pytest.raises(KeyError):
            adapter(simulator.sample(1))
        # Don't use this fixture combination for further tests.
        pytest.skip()
    elif not adapter_with_sample_weight and simulator_with_sample_weight:
        # When a weight key is present, but the adapter does not configure it
        # to be used as sample weight, no error is raised currently.
        # Don't use this fixture combination for further tests.
        pytest.skip()

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
