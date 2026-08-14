"""Tests for compositional sampling and prior score computation with adapters."""

import keras
import numpy as np

from bayesflow import CompositionalApproximator


def mock_prior_score_original_space(data_dict, time):
    """Mock prior score function that expects data in original space."""
    loc = data_dict["loc"]

    # Simple prior: N(0,1) for loc
    loc_score = -loc
    return {"loc": (1.0 - time) * loc_score}


def mock_prior_score_loc_scale(data_dict, time):
    """Prior score in the original space: loc ~ N(0, 1), scale ~ LogNormal(0, 1)."""
    loc, scale = data_dict["loc"], data_dict["scale"]

    return {
        "loc": (1.0 - time) * (-loc),
        "scale": (1.0 - time) * (-(keras.ops.log(scale) + 1.0) / scale),
    }


def test_prior_score_identity_adapter(simple_log_simulator, identity_adapter, compositional_diffusion_network):
    # Create approximator with transforming adapter
    approximator = CompositionalApproximator(
        adapter=identity_adapter,
        inference_network=compositional_diffusion_network,
    )

    # Generate test data and adapt it
    data = simple_log_simulator.sample((2,))
    adapted_data = identity_adapter(data)

    # Build approximator
    approximator.build_from_data(adapted_data)

    # Test compositional sampling
    n_datasets, n_compositional = 8, 5
    conditions = {"conditions": np.random.normal(0.0, 1.0, (n_datasets, n_compositional, 3))}
    samples = approximator.compositional_sample(
        num_samples=10, conditions=conditions, compute_prior_score=mock_prior_score_original_space, batch_size=4
    )

    assert "loc" in samples
    assert samples["loc"].shape == (n_datasets, 10, 2)


def test_prior_score_transforming_adapter(simple_log_simulator, transforming_adapter, compositional_diffusion_network):
    """An adapter with non-zero log_det_jac needs a change-of-variables correction on the prior score."""
    approximator = CompositionalApproximator(
        adapter=transforming_adapter,
        inference_network=compositional_diffusion_network,
    )

    data = simple_log_simulator.sample((2,))
    approximator.build_from_data(transforming_adapter(data))

    n_datasets, n_compositional = 4, 3
    conditions = {"conditions": np.random.normal(0.0, 1.0, (n_datasets, n_compositional, 3))}
    samples = approximator.compositional_sample(
        num_samples=5, conditions=conditions, compute_prior_score=mock_prior_score_loc_scale
    )

    assert samples["loc"].shape == (n_datasets, 5, 2)
    assert samples["scale"].shape == (n_datasets, 5, 2)
    # the log transform is inverted on the way out, so scale must be positive and finite
    assert np.all(samples["scale"] > 0.0)
    assert np.all(np.isfinite(samples["loc"]))
