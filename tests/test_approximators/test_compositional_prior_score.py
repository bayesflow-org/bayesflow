"""Tests for compositional sampling and prior score computation with adapters."""

import numpy as np

from bayesflow import ContinuousApproximator


def mock_prior_score_original_space(data_dict):
    """Mock prior score function that expects data in original space."""
    loc = data_dict["loc"]

    # Simple prior: N(0,1) for loc
    loc_score = -loc
    return {"loc": loc_score}


def test_prior_score_identity_adapter(simple_log_simulator, identity_adapter, compositional_diffusion_network):
    # Create approximator with transforming adapter
    approximator = ContinuousApproximator(
        adapter=identity_adapter,
        inference_network=compositional_diffusion_network,
    )

    # Generate test data and adapt it
    data = simple_log_simulator.sample((2,))
    adapted_data = identity_adapter(data)

    # Build approximator
    approximator.build_from_data(adapted_data)

    # Test compositional sampling
    n_datasets, n_compositional = 3, 5
    conditions = {"conditions": np.random.normal(0.0, 1.0, (n_datasets, n_compositional, 3)).astype("float32")}
    samples = approximator.compositional_sample(
        num_samples=10,
        conditions=conditions,
        compute_prior_score=mock_prior_score_original_space,
    )

    assert "loc" in samples
    assert samples["loc"].shape == (n_datasets, 10, 2)
