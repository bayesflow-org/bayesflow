import pytest
import keras
from tests.utils import check_combination_simulator_adapter


def test_approximator_sample(approximator, simulator, batch_size, adapter):
    check_combination_simulator_adapter(simulator, adapter)

    num_batches = 4
    data = simulator.sample(num_batches * batch_size)

    batch = adapter(data)
    batch = keras.tree.map_structure(keras.ops.convert_to_tensor, batch)
    batch_shapes = keras.tree.map_structure(keras.ops.shape, batch)
    approximator.build(batch_shapes)

    samples = approximator.sample(num_samples=2, conditions=data)

    assert isinstance(samples, dict)


@pytest.mark.parametrize("inference_network_type", ["flow_matching", "diffusion_model"])
@pytest.mark.parametrize("summary_network_type", ["none", "deep_set", "set_transformer", "time_series"])
@pytest.mark.parametrize("method", ["euler", "rk45", "euler_maruyama"])
def test_approximator_sample_with_integration_methods(
    inference_network_type, summary_network_type, method, simulator, adapter
):
    """Test approximator sampling with different integration methods and summary networks.

    Tests flow matching and diffusion models with different ODE/SDE solvers:
    - euler, rk45: Available for both flow matching and diffusion models
    - euler_maruyama: Only for diffusion models (stochastic)

    Also tests with different summary network types.
    """
    batch_size = 8  # Use smaller batch size for faster tests
    check_combination_simulator_adapter(simulator, adapter)

    # Skip euler_maruyama for flow matching (deterministic model)
    if inference_network_type == "flow_matching" and method == "euler_maruyama":
        pytest.skip("euler_maruyama is only available for diffusion models")

    # Create inference network based on type
    if inference_network_type == "flow_matching":
        from bayesflow.networks import FlowMatching

        inference_network = FlowMatching(
            subnet_kwargs=dict(widths=[8, 8]),
            integrate_kwargs={"steps": 10},  # Use fewer steps for faster tests
        )
    elif inference_network_type == "diffusion_model":
        from bayesflow.networks import DiffusionModel

        inference_network = DiffusionModel(
            subnet_kwargs=dict(widths=[8, 8]),
            integrate_kwargs={"steps": 10},  # Use fewer steps for faster tests
        )
    else:
        pytest.skip(f"Unsupported inference network type: {inference_network_type}")

    # Create summary network based on type
    summary_network = None
    if summary_network_type != "none":
        if summary_network_type == "deep_set":
            from bayesflow.networks import DeepSet, MLP

            summary_network = DeepSet(subnet=MLP(widths=[16, 16]))
        elif summary_network_type == "set_transformer":
            from bayesflow.networks import SetTransformer

            summary_network = SetTransformer(embed_dims=[16, 16], mlp_widths=[16, 16])
        elif summary_network_type == "time_series":
            from bayesflow.networks import TimeSeriesNetwork

            summary_network = TimeSeriesNetwork(subnet_kwargs={"widths": [16, 16]}, cell_type="lstm")
        else:
            pytest.skip(f"Unsupported summary network type: {summary_network_type}")

        # Update adapter to include summary variables if summary network is present
        from bayesflow import ContinuousApproximator

        adapter = ContinuousApproximator.build_adapter(
            inference_variables=["mean", "std"],
            summary_variables=["x"],  # Use x as summary variable for testing
        )

    # Create approximator
    from bayesflow import ContinuousApproximator

    approximator = ContinuousApproximator(
        adapter=adapter, inference_network=inference_network, summary_network=summary_network
    )

    # Generate test data
    num_batches = 2  # Use fewer batches for faster tests
    data = simulator.sample(num_batches * batch_size)

    # Build approximator
    batch = adapter(data)
    batch = keras.tree.map_structure(keras.ops.convert_to_tensor, batch)
    batch_shapes = keras.tree.map_structure(keras.ops.shape, batch)
    approximator.build(batch_shapes)

    # Test sampling with the specified method
    samples = approximator.sample(num_samples=2, conditions=data, method=method)

    # Verify results
    assert isinstance(samples, dict)
