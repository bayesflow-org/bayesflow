import pytest
from tests.utils import check_combination_simulator_adapter


@pytest.fixture()
def batch_size():
    return 8


@pytest.fixture()
def num_samples():
    return 100


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
    from bayesflow.scores import NormedDifferenceScore, QuantileScore, MultivariateNormalScore

    return PointInferenceNetwork(
        scores=dict(
            mean=NormedDifferenceScore(k=2),
            quantiles=QuantileScore(q=[0.1, 0.5, 0.9]),
            mvn=MultivariateNormalScore(),
        ),
        subnet="mlp",
        subnet_kwargs=dict(widths=(32, 32)),
    )


@pytest.fixture()
def point_inference_network_with_multiple_parametric_scores():
    from bayesflow.networks import PointInferenceNetwork
    from bayesflow.scores import MultivariateNormalScore

    return PointInferenceNetwork(
        scores=dict(
            mvn1=MultivariateNormalScore(),
            mvn2=MultivariateNormalScore(),
        ),
    )


@pytest.fixture()
def point_approximator(adapter, point_inference_network, summary_network):
    from bayesflow import PointApproximator

    return PointApproximator(
        adapter=adapter,
        inference_network=point_inference_network,
        summary_network=summary_network,
    )


@pytest.fixture()
def point_approximator_with_multiple_parametric_scores(
    adapter, point_inference_network_with_multiple_parametric_scores, summary_network
):
    from bayesflow import PointApproximator

    return PointApproximator(
        adapter=adapter,
        inference_network=point_inference_network_with_multiple_parametric_scores,
        summary_network=summary_network,
    )


@pytest.fixture(
    params=["continuous_approximator", "point_approximator", "point_approximator_with_multiple_parametric_scores"],
    scope="function",
)
def approximator(request):
    return request.getfixturevalue(request.param)


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
    check_combination_simulator_adapter(simulator, adapter)

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
