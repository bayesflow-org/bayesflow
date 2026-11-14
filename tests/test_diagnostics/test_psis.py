import sys

import numpy as np
import pytest
from scipy.stats import norm

import bayesflow as bf

arviz_stats = pytest.importorskip("arviz_stats")

# conjugate model: theta ~ N(0, 2^2), x_i ~ N(theta, 1)
PRIOR_SCALE = 2.0


def log_prior(theta):
    return norm.logpdf(theta, 0.0, PRIOR_SCALE)


def log_likelihood(theta, x):
    return norm.logpdf(x[..., None, :], theta[..., None], 1.0).sum(axis=-1)


def proposal(x, scale_factor, num_samples=2000):
    """Gaussian proposal around the analytic posterior, returns theta and its log density."""
    precision = 1 / PRIOR_SCALE**2 + x.shape[-1]
    mean = x.sum(axis=-1, keepdims=True) / precision
    scale = scale_factor / np.sqrt(precision)
    theta = np.random.normal(mean, scale, size=(x.shape[0], num_samples))
    return theta, norm.logpdf(theta, mean, scale)


def test_pareto_smooth_matches_arviz():
    """Test pareto_smooth against arviz-stats psislw."""
    log_ratios = np.random.standard_t(df=3, size=(4, 1000))

    result = bf.diagnostics.pareto_smooth(log_ratios)
    expected_log_weights, expected_k = arviz_stats.base.array_stats.psislw(-log_ratios, axis=-1)

    np.testing.assert_allclose(result["log_weights"], expected_log_weights)
    np.testing.assert_allclose(result["pareto_k"], expected_k)


def test_pareto_smooth_batch_shape():
    """Test that pareto_smooth smooths along the last axis for any batch shape."""
    log_ratios = np.random.standard_t(df=3, size=(2, 3, 1000))

    result = bf.diagnostics.pareto_smooth(log_ratios)

    assert result["log_weights"].shape == (2, 3, 1000)
    assert result["pareto_k"].shape == (2, 3)
    np.testing.assert_allclose(
        result["log_weights"][1, 2], bf.diagnostics.pareto_smooth(log_ratios[1, 2])["log_weights"]
    )


def test_pareto_smooth_missing_arviz_stats(monkeypatch):
    """Test that pareto_smooth raises an informative error without arviz-stats."""
    monkeypatch.setitem(sys.modules, "arviz_stats.base", None)

    with pytest.raises(ImportError, match="pip install arviz-stats"):
        bf.diagnostics.pareto_smooth(np.zeros(100))


def test_pareto_smooth_recovers_posterior(x):
    """Test that the weights recover the analytic posterior mean and variance."""
    theta, log_posterior = proposal(x, scale_factor=1.5)

    result = bf.diagnostics.pareto_smooth(log_prior(theta) + log_likelihood(theta, x) - log_posterior)

    weights = np.exp(result["log_weights"])
    precision = 1 / PRIOR_SCALE**2 + x.shape[-1]
    mean = (weights * theta).sum(axis=-1)
    variance = (weights * theta**2).sum(axis=-1) - mean**2

    assert result["pareto_k"].shape == (4,)
    np.testing.assert_allclose(weights.sum(axis=-1), 1.0)
    np.testing.assert_allclose(mean, x.sum(axis=-1) / precision, atol=0.05)
    np.testing.assert_allclose(variance, 1 / precision, rtol=0.15)


def test_pareto_smooth_warns_for_narrow_proposal(x, caplog):
    """Test that a proposal narrower than the posterior triggers the Pareto k warning."""
    theta, log_posterior = proposal(x, scale_factor=0.3)

    bf.diagnostics.pareto_smooth(log_prior(theta) + log_likelihood(theta, x) - log_posterior)

    assert "Pareto k" in caplog.text
