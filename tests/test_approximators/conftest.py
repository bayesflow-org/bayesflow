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
def point_approximator_with_single_parametric_score(adapter, point_inference_network, summary_network):
    from bayesflow import PointApproximator

    if "-> 'inference_conditions'" not in str(adapter) and "-> 'summary_conditions'" not in str(adapter):
        pytest.skip("point approximator does not support unconditional estimation")

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

    if "-> 'inference_conditions'" not in str(adapter) and "-> 'summary_conditions'" not in str(adapter):
        pytest.skip("point approximator does not support unconditional estimation")

    return PointApproximator(
        adapter=adapter,
        inference_network=point_inference_network_with_multiple_parametric_scores,
        summary_network=summary_network,
    )


@pytest.fixture(
    params=["point_approximator_with_single_parametric_score", "point_approximator_with_multiple_parametric_scores"]
)
def point_approximator(request):
    return request.getfixturevalue(request.param)


@pytest.fixture(
    params=[
        "continuous_approximator",
        "point_approximator_with_single_parametric_score",
        "point_approximator_with_multiple_parametric_scores",
    ],
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


@pytest.fixture()
def mean_std_summary_network():
    from tests.utils import MeanStdSummaryNetwork

    return MeanStdSummaryNetwork()


@pytest.fixture(params=["continuous_approximator", "point_approximator", "model_comparison_approximator"])
def approximator_with_summaries(request):
    from bayesflow.adapters import Adapter

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
                num_models=2, classifier_network=None, adapter=adapter, summary_network=None
            )
        case _:
            raise ValueError("Invalid param for approximator class.")


@pytest.fixture
def simple_log_simulator():
    """Create a simple simulator for testing."""
    import numpy as np
    from bayesflow.simulators import Simulator
    from bayesflow.utils.decorators import allow_batch_size
    from bayesflow.types import Shape, Tensor

    class SimpleSimulator(Simulator):
        """Simple simulator that generates mean and scale parameters."""

        @allow_batch_size
        def sample(self, batch_shape: Shape) -> dict[str, Tensor]:
            # Generate parameters in original space
            loc = np.random.normal(0.0, 1.0, size=batch_shape + (2,))  # location parameters
            scale = np.random.lognormal(0.0, 0.5, size=batch_shape + (2,))  # scale parameters > 0

            # Generate some dummy conditions
            conditions = np.random.normal(0.0, 1.0, size=batch_shape + (3,))

            return dict(
                loc=loc.astype("float32"), scale=scale.astype("float32"), conditions=conditions.astype("float32")
            )

    return SimpleSimulator()


@pytest.fixture
def identity_adapter():
    """Create an adapter that applies no transformation to the parameters."""
    from bayesflow.adapters import Adapter

    adapter = Adapter()
    adapter.to_array()
    adapter.convert_dtype("float64", "float32")

    adapter.concatenate(["loc"], into="inference_variables")
    adapter.concatenate(["conditions"], into="inference_conditions")
    adapter.keep(["inference_variables", "inference_conditions"])
    return adapter


@pytest.fixture
def transforming_adapter():
    """Create an adapter that applies log transformation to scale parameters."""
    from bayesflow.adapters import Adapter

    adapter = Adapter()
    adapter.to_array()
    adapter.convert_dtype("float64", "float32")

    # Apply log transformation to scale parameters (to make them unbounded)
    adapter.log(["scale"])

    adapter.concatenate(["scale", "loc"], into="inference_variables")
    adapter.concatenate(["conditions"], into="inference_conditions")
    adapter.keep(["inference_variables", "inference_conditions"])
    return adapter


@pytest.fixture
def compositional_diffusion_network():
    """Create a diffusion network for compositional sampling."""
    from bayesflow.networks import CompositionalDiffusionModel, MLP

    return CompositionalDiffusionModel(subnet=MLP(widths=[32, 32]))
