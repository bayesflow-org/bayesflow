import logging

import numpy as np
import pytest
from scipy.stats import norm

import bayesflow as bf
from bayesflow.types import LogProb
from bayesflow.utils import importance_resample, psis_resample


def test_public_exports():
    assert bf.types.LogProb is LogProb
    assert bf.utils.importance_sampling.psis_resample is psis_resample
    assert bf.utils.importance_sampling.importance_resample is importance_resample
    assert "LogProb" in bf.types.__all__
    assert "importance_sampling" in bf.utils.__all__


def test_importance_resample_accepts_callbacks_and_arrays():
    rng = np.random.default_rng(1)
    samples = rng.normal(size=(2, 500, 2))
    old_log_prob = norm.logpdf(samples[..., 0])
    new_log_prob = norm.logpdf(samples[..., 0], loc=0.5, scale=0.8)

    from_arrays = importance_resample(
        samples,
        old_log_prob,
        new_log_prob,
        rng=np.random.default_rng(2),
    )
    from_callbacks = importance_resample(
        samples,
        lambda value: norm.logpdf(value[..., 0]),
        lambda value: norm.logpdf(value[..., 0], loc=0.5, scale=0.8),
        rng=np.random.default_rng(2),
    )

    np.testing.assert_array_equal(from_callbacks, from_arrays)


def test_log_prob_callbacks_receive_the_sample_structure():
    rng = np.random.default_rng(17)
    samples = {
        "theta": rng.normal(size=(2, 500, 1)),
        "auxiliary": rng.normal(size=(2, 500, 3)),
    }

    result = importance_resample(
        samples,
        lambda value: norm.logpdf(value["theta"][..., 0]),
        lambda value: norm.logpdf(value["theta"][..., 0], loc=0.25),
        rng=np.random.default_rng(18),
    )

    assert result["theta"].shape == samples["theta"].shape
    assert result["auxiliary"].shape == samples["auxiliary"].shape


def test_psis_resample_accepts_callbacks_and_arrays():
    rng = np.random.default_rng(3)
    samples = rng.normal(size=(2, 1_000, 1))
    prior = norm.logpdf(samples[..., 0], scale=2.0)
    likelihood = norm.logpdf(0.5, loc=samples[..., 0])
    posterior = norm.logpdf(samples[..., 0])
    original_samples = samples.copy()
    original_log_probs = tuple(value.copy() for value in (prior, likelihood, posterior))

    from_arrays, array_diagnostics = psis_resample(
        samples,
        prior,
        likelihood,
        posterior,
        rng=np.random.default_rng(4),
        return_diagnostics=True,
    )
    from_callbacks, callback_diagnostics = psis_resample(
        samples,
        lambda value: norm.logpdf(value[..., 0], scale=2.0),
        lambda value: norm.logpdf(0.5, loc=value[..., 0]),
        lambda value: norm.logpdf(value[..., 0]),
        rng=np.random.default_rng(4),
        return_diagnostics=True,
    )

    np.testing.assert_array_equal(from_callbacks, from_arrays)
    np.testing.assert_allclose(callback_diagnostics["pareto_k"], array_diagnostics["pareto_k"])
    np.testing.assert_array_equal(samples, original_samples)
    for value, original in zip((prior, likelihood, posterior), original_log_probs):
        np.testing.assert_array_equal(value, original)


def test_mapping_samples_keep_joint_draws_and_are_reproducible():
    batch_size, num_samples = 3, 200
    draw_id = np.broadcast_to(np.arange(num_samples), (batch_size, num_samples))
    batch_id = np.broadcast_to(np.arange(batch_size)[:, None], draw_id.shape)
    encoded_id = 1_000 * batch_id + draw_id
    samples = {
        "theta": encoded_id[..., None],
        "nested": {
            "twice_theta": 2 * encoded_id[..., None],
            "vector": np.stack((encoded_id, -encoded_id), axis=-1),
        },
    }
    original = {
        "theta": samples["theta"].copy(),
        "nested": {key: value.copy() for key, value in samples["nested"].items()},
    }
    old_log_prob = np.zeros((batch_size, num_samples))
    new_log_prob = np.broadcast_to(np.linspace(-3.0, 0.0, num_samples), old_log_prob.shape).copy()

    first = importance_resample(
        samples,
        old_log_prob,
        new_log_prob,
        rng=np.random.default_rng(5),
    )
    second = importance_resample(
        samples,
        old_log_prob,
        new_log_prob,
        rng=np.random.default_rng(5),
    )

    np.testing.assert_array_equal(first["theta"], second["theta"])
    np.testing.assert_array_equal(first["nested"]["twice_theta"], 2 * first["theta"])
    np.testing.assert_array_equal(first["nested"]["vector"][..., :1], first["theta"])
    np.testing.assert_array_equal(first["nested"]["vector"][..., 1:], -first["theta"])
    np.testing.assert_array_equal(samples["theta"], original["theta"])
    for key in samples["nested"]:
        np.testing.assert_array_equal(samples["nested"][key], original["nested"][key])
    np.testing.assert_array_equal(old_log_prob, np.zeros_like(old_log_prob))
    np.testing.assert_array_equal(
        new_log_prob,
        np.broadcast_to(np.linspace(-3.0, 0.0, num_samples), new_log_prob.shape),
    )


def test_resampling_supports_multiple_batch_dimensions():
    rng = np.random.default_rng(6)
    samples = rng.normal(size=(2, 3, 400, 2))
    old_log_prob = np.zeros((2, 3, 400))
    new_log_prob = -0.5 * samples[..., 0] ** 2

    result = importance_resample(
        samples,
        old_log_prob,
        new_log_prob,
        rng=np.random.default_rng(7),
    )

    assert result.shape == samples.shape


def test_importance_resample_recovers_posterior_under_new_prior():
    rng = np.random.default_rng(8)
    observations = np.array([-1.0, 1.5])
    num_samples = 20_000

    old_prior_mean, old_prior_scale = 0.0, 1.0
    new_prior_mean, new_prior_scale = 0.5, 0.8
    likelihood_scale = 1.0

    old_variance = 1 / (1 / old_prior_scale**2 + 1 / likelihood_scale**2)
    old_mean = old_variance * (old_prior_mean / old_prior_scale**2 + observations / likelihood_scale**2)
    samples = rng.normal(
        loc=old_mean[:, None, None],
        scale=np.sqrt(old_variance),
        size=(observations.size, num_samples, 1),
    )

    shifted = importance_resample(
        samples,
        norm.logpdf(samples[..., 0], loc=old_prior_mean, scale=old_prior_scale),
        norm.logpdf(samples[..., 0], loc=new_prior_mean, scale=new_prior_scale),
        rng=np.random.default_rng(9),
    )

    expected_variance = 1 / (1 / new_prior_scale**2 + 1 / likelihood_scale**2)
    expected_mean = expected_variance * (new_prior_mean / new_prior_scale**2 + observations / likelihood_scale**2)
    np.testing.assert_allclose(shifted[..., 0].mean(axis=1), expected_mean, atol=0.025)
    np.testing.assert_allclose(shifted[..., 0].var(axis=1), expected_variance, atol=0.025)


def test_psis_resample_recovers_analytic_posterior_and_returns_diagnostics():
    rng = np.random.default_rng(10)
    observations = np.array([-0.5, 1.5])
    num_samples = 20_000
    prior_scale = 2.0

    expected_variance = 1 / (1 / prior_scale**2 + 1)
    expected_mean = expected_variance * observations
    proposal_mean = expected_mean + 0.35
    proposal_scale = 1.25 * np.sqrt(expected_variance)
    samples = rng.normal(
        loc=proposal_mean[:, None, None],
        scale=proposal_scale,
        size=(observations.size, num_samples, 1),
    )
    theta = samples[..., 0]

    corrected, diagnostics = psis_resample(
        samples,
        norm.logpdf(theta, scale=prior_scale),
        norm.logpdf(observations[:, None], loc=theta),
        norm.logpdf(theta, loc=proposal_mean[:, None], scale=proposal_scale),
        rng=np.random.default_rng(11),
        return_diagnostics=True,
    )

    np.testing.assert_allclose(corrected[..., 0].mean(axis=1), expected_mean, atol=0.035)
    np.testing.assert_allclose(corrected[..., 0].var(axis=1), expected_variance, atol=0.035)
    assert diagnostics["pareto_k"].shape == observations.shape
    assert np.all(np.isfinite(diagnostics["pareto_k"]))
    assert np.all(diagnostics["pareto_k"] < 0.7)


def test_psis_resample_logs_unreliable_weights(caplog):
    rng = np.random.default_rng(12)
    num_samples = 2_000
    samples = rng.normal(size=(1, num_samples, 1))
    heavy_tailed_weights = rng.pareto(0.5, size=(1, num_samples)) + 1

    with caplog.at_level(logging.WARNING, logger="bayesflow"):
        psis_resample(
            samples,
            np.log(heavy_tailed_weights),
            np.zeros_like(heavy_tailed_weights),
            np.zeros_like(heavy_tailed_weights),
            rng=np.random.default_rng(13),
        )
    assert "Pareto k" in caplog.text


def test_psis_pareto_k_matches_reference(caplog):
    log_ratios = np.concatenate((np.linspace(-3, 1, 45), [1.2, 1.5, 2.0, 3.0, 5.0]))

    with caplog.at_level(logging.WARNING, logger="bayesflow"):
        _, diagnostics = psis_resample(
            np.arange(log_ratios.size),
            log_ratios,
            np.zeros_like(log_ratios),
            np.zeros_like(log_ratios),
            rng=np.random.default_rng(20),
            return_diagnostics=True,
        )

    assert "Pareto k" in caplog.text
    np.testing.assert_allclose(diagnostics["pareto_k"], 0.9349972141907152)


@pytest.mark.parametrize(
    "new_log_prob",
    [
        np.full((2, 100), np.nan),
        np.full((2, 100), -np.inf),
    ],
)
def test_importance_resample_validates_log_weights(new_log_prob):
    samples = np.zeros((2, 100, 1))
    with pytest.raises(ValueError):
        importance_resample(
            samples,
            np.zeros((2, 100)),
            new_log_prob,
            rng=np.random.default_rng(14),
        )


def test_psis_resample_validates_log_prob_shapes():
    samples = np.zeros((2, 100, 1))

    with pytest.raises(ValueError):
        psis_resample(
            samples,
            np.zeros((2, 100)),
            np.zeros((2, 99)),
            np.zeros((2, 100)),
            rng=np.random.default_rng(15),
        )


def test_importance_resample_validates_callback_output_shape():
    samples = np.zeros((2, 100, 1))

    with pytest.raises(ValueError):
        importance_resample(
            samples,
            lambda value: np.zeros(value.shape[:2]),
            lambda value: np.zeros(value.shape[1]),
            rng=np.random.default_rng(19),
        )


def test_psis_resample_requires_enough_draws():
    samples = np.zeros((1, 4, 1))
    log_prob = np.zeros((1, 4))

    with pytest.raises(ValueError):
        psis_resample(
            samples,
            log_prob,
            log_prob,
            log_prob,
            rng=np.random.default_rng(16),
        )
