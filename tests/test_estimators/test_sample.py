import pytest
import keras
from tests.utils import check_combination_simulator_adapter

from bayesflow import OnlineDataset, EnsembleDataset, EnsembleEstimator


def test_estimator_sample(estimator, simulator, batch_size, adapter):
    check_combination_simulator_adapter(simulator, adapter)

    data = simulator.sample(batch_size)

    train_dataset = OnlineDataset(simulator=simulator, adapter=adapter, num_batches=4, batch_size=batch_size)
    if isinstance(estimator, EnsembleEstimator):
        train_dataset = EnsembleDataset(train_dataset, member_names=list(estimator.estimators.keys()))
    batch = train_dataset[0]

    print(keras.tree.map_structure(keras.ops.shape, batch))

    estimator.build_from_data(batch)

    num_samples = 11
    if estimator.has_distribution:
        samples = estimator.sample(conditions=data, num_samples=num_samples)
        assert isinstance(samples, dict)
        if "inference_conditions" in batch.keys():
            expected_shape = (batch_size, num_samples)
        else:
            expected_shape = (num_samples,)
        for v in samples.values():
            assert v.shape[: len(expected_shape)] == expected_shape
    else:
        with pytest.raises(ValueError):
            estimator.log_prob(data)


@pytest.mark.parametrize("inference_network_type", ["flow_matching", "diffusion_model"])
@pytest.mark.parametrize("summary_network_type", ["none", "deep_set", "set_transformer", "time_series"])
@pytest.mark.parametrize("method", ["euler", "rk45", "euler_maruyama"])
def test_estimator_sample_with_integration_methods(
    inference_network_type, summary_network_type, method, simulator, adapter
):
    """Test estimator sampling with different integration methods and summary networks.

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
        from bayesflow import BasicEstimator

        adapter = BasicEstimator.build_adapter(
            inference_variables=["mean", "std"],
            summary_variables=["x"],  # Use x as summary variable for testing
        )

    # Create estimator
    from bayesflow import BasicEstimator

    estimator = BasicEstimator(adapter=adapter, inference_network=inference_network, summary_network=summary_network)

    # Generate test data
    num_batches = 2  # Use fewer batches for faster tests
    data = simulator.sample((num_batches * batch_size,))

    # Build estimator
    batch = adapter(data)
    batch = keras.tree.map_structure(keras.ops.convert_to_tensor, batch)
    batch_shapes = keras.tree.map_structure(keras.ops.shape, batch)
    estimator.build(batch_shapes)

    # Test sampling with the specified method
    samples = estimator.sample(num_samples=2, conditions=data, method=method)

    # Verify results
    assert isinstance(samples, dict)
