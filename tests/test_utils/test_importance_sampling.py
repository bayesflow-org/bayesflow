import logging

import numpy as np
import pytest
from scipy.stats import norm

import bayesflow as bf
from bayesflow.types import LogProb
from bayesflow.utils import importance_resample
from bayesflow.workflows.workflow import Workflow


def test_public_exports():
    assert bf.types.LogProb is LogProb
    assert bf.utils.importance_sampling.importance_resample is importance_resample
    assert "LogProb" in bf.types.__all__
    assert "importance_sampling" in bf.utils.__all__
    assert "importance_resample" in bf.utils.__all__
    assert not hasattr(bf.utils, "psis_resample")


def test_importance_resample_accepts_callbacks_and_arrays():
    rng = np.random.default_rng(1)
    samples = rng.normal(size=(2, 500, 2))
    log_ratio = norm.logpdf(samples[..., 0], loc=0.5, scale=0.8) - norm.logpdf(samples[..., 0])

    from_arrays = importance_resample(samples, log_ratio, method="is", rng=np.random.default_rng(2))
    from_callbacks = importance_resample(
        samples,
        lambda value: norm.logpdf(value[..., 0], loc=0.5, scale=0.8) - norm.logpdf(value[..., 0]),
        method="is",
        rng=np.random.default_rng(2),
    )

    np.testing.assert_array_equal(from_callbacks, from_arrays)


def test_log_ratio_callback_receives_the_sample_structure():
    rng = np.random.default_rng(17)
    samples = {
        "theta": rng.normal(size=(2, 500, 1)),
        "auxiliary": rng.normal(size=(2, 500, 3)),
    }

    result = importance_resample(
        samples,
        lambda value: norm.logpdf(value["theta"][..., 0], loc=0.25) - norm.logpdf(value["theta"][..., 0]),
        method="is",
        rng=np.random.default_rng(18),
    )

    assert result["theta"].shape == samples["theta"].shape
    assert result["auxiliary"].shape == samples["auxiliary"].shape


def test_psis_accepts_callbacks_and_arrays():
    rng = np.random.default_rng(3)
    samples = rng.normal(size=(2, 1_000, 1))
    log_ratio = (
        norm.logpdf(samples[..., 0], scale=2.0) + norm.logpdf(0.5, loc=samples[..., 0]) - norm.logpdf(samples[..., 0])
    )
    original_samples = samples.copy()
    original_log_ratio = log_ratio.copy()

    from_arrays, array_diagnostics = importance_resample(
        samples,
        log_ratio,
        method="psis",
        rng=np.random.default_rng(4),
        return_diagnostics=True,
    )
    from_callbacks, callback_diagnostics = importance_resample(
        samples,
        lambda value: (
            norm.logpdf(value[..., 0], scale=2.0) + norm.logpdf(0.5, loc=value[..., 0]) - norm.logpdf(value[..., 0])
        ),
        method="psis",
        rng=np.random.default_rng(4),
        return_diagnostics=True,
    )

    np.testing.assert_array_equal(from_callbacks, from_arrays)
    np.testing.assert_allclose(callback_diagnostics["pareto_k"], array_diagnostics["pareto_k"])
    np.testing.assert_array_equal(samples, original_samples)
    np.testing.assert_array_equal(log_ratio, original_log_ratio)


def test_psis_accepts_integer_log_ratios():
    samples = np.arange(100)
    log_ratio = np.arange(100) // 10

    from_integers, integer_diagnostics = importance_resample(
        samples,
        log_ratio,
        rng=np.random.default_rng(5),
        return_diagnostics=True,
    )
    from_floats, float_diagnostics = importance_resample(
        samples,
        log_ratio.astype(float),
        rng=np.random.default_rng(5),
        return_diagnostics=True,
    )

    np.testing.assert_array_equal(from_integers, from_floats)
    np.testing.assert_array_equal(integer_diagnostics["pareto_k"], float_diagnostics["pareto_k"])


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
    log_ratio = np.broadcast_to(np.linspace(-3.0, 0.0, num_samples), draw_id.shape).copy()

    first = importance_resample(samples, log_ratio, method="is", rng=np.random.default_rng(5))
    second = importance_resample(samples, log_ratio, method="is", rng=np.random.default_rng(5))

    np.testing.assert_array_equal(first["theta"], second["theta"])
    np.testing.assert_array_equal(first["nested"]["twice_theta"], 2 * first["theta"])
    np.testing.assert_array_equal(first["nested"]["vector"][..., :1], first["theta"])
    np.testing.assert_array_equal(first["nested"]["vector"][..., 1:], -first["theta"])
    np.testing.assert_array_equal(samples["theta"], original["theta"])
    for key in samples["nested"]:
        np.testing.assert_array_equal(samples["nested"][key], original["nested"][key])
    np.testing.assert_array_equal(
        log_ratio,
        np.broadcast_to(np.linspace(-3.0, 0.0, num_samples), log_ratio.shape),
    )


def test_resampling_supports_multiple_batch_dimensions():
    rng = np.random.default_rng(6)
    samples = rng.normal(size=(2, 3, 400, 2))

    result = importance_resample(
        samples,
        -0.5 * samples[..., 0] ** 2,
        method="is",
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
    log_ratio = norm.logpdf(samples[..., 0], loc=new_prior_mean, scale=new_prior_scale) - norm.logpdf(
        samples[..., 0], loc=old_prior_mean, scale=old_prior_scale
    )

    shifted = importance_resample(samples, log_ratio, method="is", rng=np.random.default_rng(9))

    expected_variance = 1 / (1 / new_prior_scale**2 + 1 / likelihood_scale**2)
    expected_mean = expected_variance * (new_prior_mean / new_prior_scale**2 + observations / likelihood_scale**2)
    np.testing.assert_allclose(shifted[..., 0].mean(axis=1), expected_mean, atol=0.025)
    np.testing.assert_allclose(shifted[..., 0].var(axis=1), expected_variance, atol=0.025)


def test_psis_recovers_analytic_posterior_and_returns_diagnostics():
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
    log_ratio = (
        norm.logpdf(theta, scale=prior_scale)
        + norm.logpdf(observations[:, None], loc=theta)
        - norm.logpdf(theta, loc=proposal_mean[:, None], scale=proposal_scale)
    )

    corrected, diagnostics = importance_resample(
        samples,
        log_ratio,
        rng=np.random.default_rng(11),
        return_diagnostics=True,
    )

    np.testing.assert_allclose(corrected[..., 0].mean(axis=1), expected_mean, atol=0.035)
    np.testing.assert_allclose(corrected[..., 0].var(axis=1), expected_variance, atol=0.035)
    assert diagnostics["pareto_k"].shape == observations.shape
    assert np.all(np.isfinite(diagnostics["pareto_k"]))
    assert np.all(diagnostics["pareto_k"] < 0.7)


def test_psis_logs_unreliable_weights(caplog):
    rng = np.random.default_rng(12)
    samples = rng.normal(size=(1, 2_000, 1))
    log_ratio = np.log(rng.pareto(0.5, size=(1, 2_000)) + 1)

    with caplog.at_level(logging.WARNING, logger="bayesflow"):
        importance_resample(samples, log_ratio, rng=np.random.default_rng(13))
    assert "Pareto k" in caplog.text


def test_psis_pareto_k_matches_reference(caplog):
    log_ratio = np.concatenate((np.linspace(-3, 1, 45), [1.2, 1.5, 2.0, 3.0, 5.0]))

    with caplog.at_level(logging.WARNING, logger="bayesflow"):
        _, diagnostics = importance_resample(
            np.arange(log_ratio.size),
            log_ratio,
            rng=np.random.default_rng(20),
            return_diagnostics=True,
        )

    assert "Pareto k" in caplog.text
    np.testing.assert_allclose(diagnostics["pareto_k"], 0.9349972141907152)


@pytest.mark.parametrize(
    "log_ratio",
    [
        np.full((2, 100), np.nan),
        np.full((2, 100), np.inf),
        np.full((2, 100), -np.inf),
    ],
)
def test_importance_resample_validates_log_ratio(log_ratio):
    with pytest.raises(ValueError):
        importance_resample(
            np.zeros((2, 100, 1)),
            log_ratio,
            method="is",
            rng=np.random.default_rng(14),
        )


def test_importance_resample_rejects_unknown_method():
    with pytest.raises(ValueError, match="method"):
        importance_resample(
            np.zeros((1, 100, 1)),
            np.zeros((1, 100)),
            method="unknown",
        )


def test_psis_requires_enough_draws():
    with pytest.raises(ValueError):
        importance_resample(
            np.zeros((1, 4, 1)),
            np.zeros((1, 4)),
            rng=np.random.default_rng(16),
        )


def test_is_supports_few_draws_and_returns_empty_diagnostics():
    samples = np.arange(4)[None, :, None]
    result, diagnostics = importance_resample(
        samples,
        np.zeros((1, 4)),
        method="is",
        rng=np.random.default_rng(19),
        return_diagnostics=True,
    )

    assert result.shape == samples.shape
    assert diagnostics == {}


def test_basic_workflow_importance_resample_matches_utility():
    samples = np.arange(100)[None, :, None]
    log_ratio = np.linspace(-2.0, 0.0, 100)[None, :]
    workflow = object.__new__(bf.BasicWorkflow)

    expected = importance_resample(samples, log_ratio, method="is", rng=np.random.default_rng(21))
    actual = workflow.importance_resample(samples, log_ratio, method="is", rng=np.random.default_rng(21))

    np.testing.assert_array_equal(actual, expected)


def test_importance_resample_is_not_defined_on_base_workflow():
    assert not hasattr(Workflow, "importance_resample")
