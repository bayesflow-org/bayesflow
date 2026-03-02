import pytest
import keras


def test_sample(continuous_approximator, simulator, batch_size, adapter):
    num_batches = 4
    data = simulator.sample((num_batches * batch_size,))

    batch = adapter(data)
    batch = keras.tree.map_structure(keras.ops.convert_to_tensor, batch)
    batch_shapes = keras.tree.map_structure(keras.ops.shape, batch)
    continuous_approximator.build(batch_shapes)

    samples = continuous_approximator.sample(num_samples=2, conditions=data)

    assert isinstance(samples, dict)


def _make_adapter(with_summary=False):
    """Build a minimal adapter for integration method tests."""
    from bayesflow import ContinuousApproximator

    if with_summary:
        return ContinuousApproximator.build_adapter(
            inference_variables=["mean", "std"],
            summary_variables=["x"],
        )
    return ContinuousApproximator.build_adapter(
        inference_variables=["mean", "std"],
        inference_conditions=["x"],
    )


def _make_inference_network(network_type):
    """Build a minimal inference network for integration method tests."""
    if network_type == "flow_matching":
        from bayesflow.networks import FlowMatching

        return FlowMatching(subnet_kwargs=dict(widths=[8, 8]), integrate_kwargs={"steps": 10})
    elif network_type == "diffusion_model":
        from bayesflow.networks import DiffusionModel

        return DiffusionModel(subnet_kwargs=dict(widths=[8, 8]), integrate_kwargs={"steps": 10})
    raise ValueError(f"Unknown network type: {network_type}")


def _build_and_sample(inference_network, summary_network, adapter, method):
    """Helper: build an approximator, generate data, sample."""
    from bayesflow import ContinuousApproximator
    from tests.utils.normal_simulator import NormalSimulator

    approximator = ContinuousApproximator(
        adapter=adapter,
        inference_network=inference_network,
        summary_network=summary_network,
    )

    simulator = NormalSimulator()
    data = simulator.sample((16,))

    batch = adapter(data)
    batch = keras.tree.map_structure(keras.ops.convert_to_tensor, batch)
    batch_shapes = keras.tree.map_structure(keras.ops.shape, batch)
    approximator.build(batch_shapes)

    samples = approximator.sample(num_samples=2, conditions=data, method=method)
    assert isinstance(samples, dict)


# --- Test integration methods (no summary network) --------------------------
# Summary network is orthogonal to integration method, so test them separately.
@pytest.mark.parametrize("inference_network_type", ["flow_matching", "diffusion_model"])
@pytest.mark.parametrize("method", ["euler", "rk45", "euler_maruyama"])
def test_integration_methods(inference_network_type, method):
    """Test ODE/SDE solvers for flow matching and diffusion models."""
    if inference_network_type == "flow_matching" and method == "euler_maruyama":
        pytest.skip("euler_maruyama is only available for diffusion models")

    inference_network = _make_inference_network(inference_network_type)
    adapter = _make_adapter(with_summary=False)
    _build_and_sample(inference_network, None, adapter, method)


# --- Test summary network compatibility (single integration method) ----------
@pytest.mark.parametrize("inference_network_type", ["flow_matching", "diffusion_model"])
@pytest.mark.parametrize("summary_network_type", ["deep_set", "set_transformer", "time_series"])
def test_summary_network_compatibility(inference_network_type, summary_network_type):
    """Test that each summary network works with continuous inference networks."""
    if summary_network_type == "deep_set":
        from bayesflow.networks import DeepSet, MLP

        summary_network = DeepSet(subnet=MLP(widths=[16, 16]))
    elif summary_network_type == "set_transformer":
        from bayesflow.networks import SetTransformer

        summary_network = SetTransformer(embed_dims=[16, 16], mlp_widths=[16, 16])
    elif summary_network_type == "time_series":
        from bayesflow.networks import TimeSeriesNetwork

        summary_network = TimeSeriesNetwork(subnet_kwargs={"widths": [16, 16]}, cell_type="lstm")

    inference_network = _make_inference_network(inference_network_type)
    adapter = _make_adapter(with_summary=True)
    _build_and_sample(inference_network, summary_network, adapter, method="euler")
