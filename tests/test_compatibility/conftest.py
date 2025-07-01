import pytest
from pathlib import Path


@pytest.fixture(autouse=True, scope="session")
def mode(request):
    mode = request.config.getoption("--mode")
    if not mode:
        return "save"
    return mode


@pytest.fixture(autouse=True, scope="session")
def data_dir(request, tmp_path_factory):
    # read config option to detect "unset" scenario
    mode = request.config.getoption("--mode")
    path = request.config.getoption("--data-path")
    if not mode:
        # if mode is unset, save and load from a temporary directory
        return Path(tmp_path_factory.mktemp("_compatibility_data"))
    elif not path:
        pytest.exit(reason="Please provide the --data-path argument for model saving/loading.")
    elif mode == "load":
        path = Path(path)
        if not path.exists():
            pytest.exit(reason=f"Load path '{path}' does not exist. Please specify a valid load path", returncode=1)
    return path


# reduce number of test configurations
@pytest.fixture(params=[None, 3])
def conditions_size(request):
    return request.param


@pytest.fixture(params=[1, 2])
def summary_dim(request):
    return request.param


@pytest.fixture(params=[4])
def feature_size(request):
    return request.param


# Generic fixtures for use as input to the tested classes.
# The classes to test are constructed in the respective subdirectories, to allow for more thorough configuation.
@pytest.fixture(params=[None, "all"])
def standardize(request):
    return request.param


@pytest.fixture()
def adapter(request):
    import bayesflow as bf

    match request.param:
        case "summary":
            return bf.Adapter.create_default("parameters").rename("observables", "summary_variables")
        case "direct":
            return bf.Adapter.create_default("parameters").rename("observables", "inference_conditions")
        case "default":
            return bf.Adapter.create_default("parameters")
        case "empty":
            return bf.Adapter()
        case None:
            return None
        case _:
            raise ValueError(f"Invalid request parameter for adapter: {request.param}")


@pytest.fixture(params=["coupling_flow", "flow_matching"])
def inference_network(request):
    match request.param:
        case "coupling_flow":
            from bayesflow.networks import CouplingFlow

            return CouplingFlow(depth=2)

        case "flow_matching":
            from bayesflow.networks import FlowMatching

            return FlowMatching(subnet_kwargs=dict(widths=(32, 32)), use_optimal_transport=False)

        case None:
            return None

        case _:
            raise ValueError(f"Invalid request parameter for inference_network: {request.param}")


@pytest.fixture(params=["time_series_transformer", "fusion_transformer", "time_series_network", "custom"])
def summary_network(request):
    match request.param:
        case "time_series_transformer":
            from bayesflow.networks import TimeSeriesTransformer

            return TimeSeriesTransformer(embed_dims=(8, 8), mlp_widths=(16, 8), mlp_depths=(1, 1))

        case "fusion_transformer":
            from bayesflow.networks import FusionTransformer

            return FusionTransformer(
                embed_dims=(8, 8), mlp_widths=(8, 16), mlp_depths=(2, 1), template_dim=8, bidirectional=False
            )

        case "time_series_network":
            from bayesflow.networks import TimeSeriesNetwork

            return TimeSeriesNetwork(filters=4, skip_steps=2)

        case "deep_set":
            from bayesflow.networks import DeepSet

            return DeepSet(summary_dim=2, depth=1)

        case "custom":
            from bayesflow.networks import SummaryNetwork
            from bayesflow.utils.serialization import serializable
            import keras

            @serializable("test", disable_module_check=True)
            class Custom(SummaryNetwork):
                def __init__(self, **kwargs):
                    super().__init__(**kwargs)
                    self.inner = keras.Sequential([keras.layers.LSTM(8), keras.layers.Dense(4)])

                def call(self, x, **kwargs):
                    return self.inner(x, training=kwargs.get("stage") == "training")

            return Custom()

        case "flatten":
            # very simple summary network for fast training
            from bayesflow.networks import SummaryNetwork
            from bayesflow.utils.serialization import serializable
            import keras

            @serializable("test", disable_module_check=True)
            class FlattenSummaryNetwork(SummaryNetwork):
                def __init__(self, **kwargs):
                    super().__init__(**kwargs)
                    self.inner = keras.layers.Flatten()

                def call(self, x, **kwargs):
                    return self.inner(x, training=kwargs.get("stage") == "training")

            return FlattenSummaryNetwork()

        case "fusion_network":
            from bayesflow.networks import FusionNetwork, DeepSet

            return FusionNetwork({"a": DeepSet(), "b": keras.layers.Flatten()}, head=keras.layers.Dense(2))
        case None:
            return None
        case _:
            raise ValueError(f"Invalid request parameter for summary_network: {request.param}")


@pytest.fixture(params=["sir", "fusion"])
def simulator(request):
    match request.param:
        case "sir":
            from bayesflow.simulators import SIR

            return SIR()
        case "lotka_volterra":
            from bayesflow.simulators import LotkaVolterra

            return LotkaVolterra()

        case "two_moons":
            from bayesflow.simulators import TwoMoons

            return TwoMoons()
        case "normal":
            from tests.utils.normal_simulator import NormalSimulator

            return NormalSimulator()
        case "fusion":
            from bayesflow.simulators import Simulator
            from bayesflow.types import Shape, Tensor
            from bayesflow.utils.decorators import allow_batch_size
            import numpy as np

            class FusionSimulator(Simulator):
                @allow_batch_size
                def sample(self, batch_shape: Shape, num_observations: int = 4) -> dict[str, Tensor]:
                    mean = np.random.normal(0.0, 0.1, size=batch_shape + (2,))
                    noise = np.random.standard_normal(batch_shape + (num_observations, 2))

                    x = mean[:, None] + noise

                    return dict(mean=mean, a=x, b=x)

            return FusionSimulator()
        case None:
            return None
        case _:
            raise ValueError(f"Invalid request parameter for simulator: {request.param}")
