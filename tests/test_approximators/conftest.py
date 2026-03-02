import pytest
from tests.utils import check_combination_simulator_adapter


@pytest.fixture(autouse=True)
def _validate_simulator_adapter_combination(request):
    """Skip invalid simulator+adapter combinations early.

    Automatically applied to all tests in test_approximators/ that request
    both ``simulator`` and ``adapter`` fixtures, so individual tests don't
    need to call ``check_combination_simulator_adapter`` manually.
    """
    if "simulator" in request.fixturenames and "adapter" in request.fixturenames:
        check_combination_simulator_adapter(
            request.getfixturevalue("simulator"),
            request.getfixturevalue("adapter"),
        )


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


@pytest.fixture()
def adapter_unconditional():
    from bayesflow import ContinuousApproximator

    return ContinuousApproximator.build_adapter(
        inference_variables=["mean", "std"],
    )


@pytest.fixture(params=["adapter_unconditional", "adapter_without_sample_weight", "adapter_with_sample_weight"])
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


@pytest.fixture()
def mean_std_summary_network():
    from tests.utils import MeanStdSummaryNetwork

    return MeanStdSummaryNetwork()


@pytest.fixture(params=["continuous_approximator", "point_approximator", "model_comparison_approximator"])
def approximator_with_summaries(request):
    from bayesflow.adapters import Adapter
    from bayesflow.networks import MLP

    adapter = Adapter()
    match request.param:
        case "continuous_approximator":
            from bayesflow.approximators import ContinuousApproximator

            return ContinuousApproximator(adapter=adapter, inference_network=None, summary_network=None)
        case "point_approximator":
            from bayesflow.approximators import PointApproximator

            return PointApproximator(adapter=adapter, inference_network=None, summary_network=None)
        case "model_comparison_approximator":
            from bayesflow.approximators import ModelComparisonApproximator

            return ModelComparisonApproximator(
                num_models=2, classifier_network=MLP(widths=(8, 8)), adapter=adapter, summary_network=None
            )
        case _:
            raise ValueError("Invalid param for approximator class.")
