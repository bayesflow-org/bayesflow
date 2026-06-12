import pytest

import keras

import bayesflow as bf
from bayesflow.networks import CouplingFlow
from bayesflow.utils.serialization import serializable


@pytest.fixture(params=["coupling_flow", "flow_matching", "diffusion_model", "consistency_model"])
def inference_network(request):
    if request.param == "coupling_flow":
        from bayesflow.networks import CouplingFlow

        return CouplingFlow(depth=1, subnet_kwargs=dict(widths=(8,)))

    elif request.param == "flow_matching":
        from bayesflow.networks import FlowMatching

        return FlowMatching(subnet_kwargs=dict(widths=(8,)), use_optimal_transport=False)

    elif request.param == "diffusion_model":
        from bayesflow.networks import DiffusionModel

        return DiffusionModel(subnet_kwargs=dict(widths=(8,)))

    elif request.param == "consistency_model":
        from bayesflow.networks import ConsistencyModel

        return ConsistencyModel(subnet_kwargs=dict(widths=(8,)), total_steps=10)


@pytest.fixture(params=["time_series_transformer", "fusion_transformer", "time_series_network", "custom"])
def summary_network(request):
    if request.param == "time_series_transformer":
        from bayesflow.networks import TimeSeriesTransformer

        return TimeSeriesTransformer(embed_dims=(4, 4), mlp_widths=(8, 4), mlp_depths=(1, 1))

    elif request.param == "fusion_transformer":
        from bayesflow.networks import FusionTransformer

        return FusionTransformer(
            embed_dims=(4, 4), mlp_widths=(4, 8), mlp_depths=(1, 1), template_dim=4, bidirectional=False
        )

    elif request.param == "time_series_network":
        from bayesflow.networks import TimeSeriesNetwork

        return TimeSeriesNetwork(filters=4, skip_steps=2)

    elif request.param == "custom":
        from bayesflow.networks import SummaryNetwork

        @serializable("test", disable_module_check=True)
        class Custom(SummaryNetwork):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.inner = keras.Sequential([keras.layers.LSTM(8), keras.layers.Dense(4)])

            def call(self, x, **kwargs):
                return self.inner(x, training=kwargs.get("stage") == "training")

        return Custom()


@pytest.fixture
def tiny_workflow(tmp_path):
    """Trained minimal workflow with a full .keras checkpoint."""
    workflow = bf.BasicWorkflow(
        inference_network=CouplingFlow(depth=1, subnet_kwargs=dict(widths=(8,))),
        inference_variables=["parameters"],
        simulator=bf.simulators.TwoMoons(),
        checkpoint_filepath=str(tmp_path),
        checkpoint_name="model",
        save_weights_only=False,
    )
    workflow.fit_online(epochs=1, batch_size=4, num_batches_per_epoch=1, verbose=0)
    return workflow


@pytest.fixture
def tiny_workflow_weights_only(tmp_path):
    """Trained minimal workflow with a weights-only .weights.h5 checkpoint."""
    workflow = bf.BasicWorkflow(
        inference_network=CouplingFlow(depth=1, subnet_kwargs=dict(widths=(8,))),
        inference_variables=["parameters"],
        simulator=bf.simulators.TwoMoons(),
        checkpoint_filepath=str(tmp_path),
        checkpoint_name="model",
        save_weights_only=True,
    )
    workflow.fit_online(epochs=1, batch_size=4, num_batches_per_epoch=1, verbose=0)
    return workflow


@pytest.fixture
def fusion_inference_network():
    from bayesflow.networks import CouplingFlow

    return CouplingFlow()


@pytest.fixture
def fusion_summary_network():
    from bayesflow.networks import FusionNetwork, DeepSet

    return FusionNetwork({"a": DeepSet(), "b": keras.layers.Flatten()}, head=keras.layers.Dense(2))


@pytest.fixture
def fusion_simulator():
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


@pytest.fixture
def fusion_adapter():
    from bayesflow import Adapter

    return Adapter.create_default(["mean"]).group(["a", "b"], "summary_variables")
