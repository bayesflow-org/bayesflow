"""Tests for compositional sampling and prior score computation with adapters."""

import numpy as np
import keras

from bayesflow import ContinuousApproximator
from bayesflow.utils import expand_right_as


def mock_prior_score_original_space(data_dict):
    """Mock prior score function that expects data in original (loc, scale) space."""
    # The function receives data in the same format the compute_prior_score_pre creates
    # after running the inverse adapter
    loc = data_dict["loc"]
    scale = data_dict["scale"]

    # Simple prior: N(0,1) for loc, LogNormal(0,0.5) for scale
    loc_score = -loc
    scale_score = -1.0 / scale - np.log(scale) / (0.25 * scale)

    return {"loc": loc_score, "scale": scale_score}


def test_prior_score_transforming_adapter(simple_log_simulator, transforming_adapter, diffusion_network):
    """Test that prior scores work correctly with transforming adapter (log transformation)."""

    # Create approximator with transforming adapter
    approximator = ContinuousApproximator(
        adapter=transforming_adapter,
        inference_network=diffusion_network,
    )

    # Generate test data and adapt it
    data = simple_log_simulator.sample((2,))
    adapted_data = transforming_adapter(data)

    # Build approximator
    approximator.build_from_data(adapted_data)

    # Test compositional sampling
    n_datasets, n_compositional = 3, 5
    conditions = {"conditions": np.random.normal(0.0, 1.0, (n_datasets, n_compositional, 3)).astype("float32")}

    # This should work - the compute_prior_score_pre function should handle the inverse transformation
    samples = approximator.compositional_sample(
        num_samples=10,
        conditions=conditions,
        compute_prior_score=mock_prior_score_original_space,
    )

    assert "loc" in samples
    assert "scale" in samples
    assert samples["loc"].shape == (n_datasets, 10, 2)
    assert samples["scale"].shape == (n_datasets, 10, 2)


def test_prior_score_jacobian_correction(simple_log_simulator, transforming_adapter, diffusion_network):
    """Test that Jacobian correction is applied correctly in compute_prior_score_pre."""

    # Create approximator with transforming adapter
    approximator = ContinuousApproximator(
        adapter=transforming_adapter, inference_network=diffusion_network, standardize=[]
    )

    # Build with dummy data
    dummy_data_dict = simple_log_simulator.sample((1,))
    adapted_dummy_data = transforming_adapter(dummy_data_dict)
    approximator.build_from_data(adapted_dummy_data)

    # Get the internal compute_prior_score_pre function
    def get_compute_prior_score_pre():
        def compute_prior_score_pre(_samples):
            if "inference_variables" in approximator.standardize:
                _samples, log_det_jac_standardize = approximator.standardize_layers["inference_variables"](
                    _samples, forward=False, log_det_jac=True
                )
            else:
                log_det_jac_standardize = keras.ops.cast(0.0, dtype="float32")

            _samples = keras.tree.map_structure(keras.ops.convert_to_numpy, {"inference_variables": _samples})
            adapted_samples, log_det_jac = approximator.adapter(_samples, inverse=True, strict=False, log_det_jac=True)

            prior_score = mock_prior_score_original_space(adapted_samples)
            for key in adapted_samples:
                if isinstance(prior_score[key], np.ndarray):
                    prior_score[key] = prior_score[key].astype("float32")
                if len(log_det_jac) > 0 and key in log_det_jac:
                    prior_score[key] -= expand_right_as(log_det_jac[key], prior_score[key])

            prior_score = keras.tree.map_structure(keras.ops.convert_to_tensor, prior_score)
            out = keras.ops.concatenate(list(prior_score.values()), axis=-1)
            return out - keras.ops.expand_dims(log_det_jac_standardize, axis=-1)

        return compute_prior_score_pre

    compute_prior_score_pre = get_compute_prior_score_pre()

    # Test with a known transformation
    y_samples = adapted_dummy_data["inference_variables"]
    scores = compute_prior_score_pre(y_samples)
    scores_np = keras.ops.convert_to_numpy(scores)[0]  # Remove batch dimension

    # With Jacobian correction: score_transformed = score_original - log|J|
    old_scores = mock_prior_score_original_space(dummy_data_dict)
    det_jac_scale = y_samples[0, 2:].sum()
    expected_scores = np.array([old_scores["loc"][0], old_scores["scale"][0] - det_jac_scale]).flatten()

    # Check that scores are reasonably close
    np.testing.assert_allclose(scores_np, expected_scores, rtol=1e-5, atol=1e-6)
