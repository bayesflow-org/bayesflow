import keras
import numpy as np
import pytest
from scipy.stats import binom

import bayesflow as bf


def num_variables(x: dict):
    return sum(arr.shape[-1] for arr in x.values())


def test_metric_calibration_error(random_estimates, random_targets, var_names):
    # basic functionality: automatic variable names
    out = bf.diagnostics.metrics.calibration_error(random_estimates, random_targets)
    assert list(out.keys()) == ["values", "metric_name", "variable_names"]
    assert out["values"].shape == (num_variables(random_estimates),)
    assert out["metric_name"] == "Calibration Error"
    assert out["variable_names"] == ["beta_0", "beta_1", "sigma"]

    # user specified variable names
    out = bf.diagnostics.metrics.calibration_error(
        estimates=random_estimates,
        targets=random_targets,
        variable_names=var_names,
    )
    assert out["variable_names"] == var_names

    # user-specifed keys and scalar variable
    out = bf.diagnostics.metrics.calibration_error(
        estimates=random_estimates,
        targets=random_targets,
        variable_keys="sigma",
    )
    assert out["values"].shape == (random_estimates["sigma"].shape[-1],)
    assert out["variable_names"] == ["sigma"]


def test_posterior_contraction(random_estimates, random_targets):
    # basic functionality: automatic variable names
    out = bf.diagnostics.metrics.posterior_contraction(random_estimates, random_targets)
    assert list(out.keys()) == ["values", "metric_name", "variable_names"]
    assert out["values"].shape == (num_variables(random_estimates),)
    assert out["metric_name"] == "Posterior Contraction"
    assert out["variable_names"] == ["beta_0", "beta_1", "sigma"]


def test_root_mean_squared_error(random_estimates, random_targets):
    # basic functionality: automatic variable names
    out = bf.diagnostics.metrics.root_mean_squared_error(random_estimates, random_targets)
    assert list(out.keys()) == ["values", "metric_name", "variable_names"]
    assert out["values"].shape == (num_variables(random_estimates),)
    assert out["metric_name"] == "NRMSE"
    assert out["variable_names"] == ["beta_0", "beta_1", "sigma"]


def test_classifier_two_sample_test(random_samples_a, random_samples_b):
    metric = bf.diagnostics.metrics.classifier_two_sample_test(estimates=random_samples_a, targets=random_samples_a)
    assert 0.55 > metric > 0.45

    metric = bf.diagnostics.metrics.classifier_two_sample_test(estimates=random_samples_a, targets=random_samples_b)
    assert metric > 0.55


def test_expected_calibration_error(pred_models, true_models, model_names):
    out = bf.diagnostics.metrics.expected_calibration_error(pred_models, true_models, model_names=model_names)
    assert list(out.keys()) == ["values", "metric_name", "model_names"]
    assert out["values"].shape == (pred_models.shape[-1],)
    assert out["metric_name"] == "Expected Calibration Error"
    assert out["model_names"] == [r"$\mathcal{M}_0$", r"$\mathcal{M}_1$", r"$\mathcal{M}_2$"]

    # returns probs?
    out = bf.diagnostics.metrics.expected_calibration_error(pred_models, true_models, return_probs=True)
    assert list(out.keys()) == ["values", "metric_name", "model_names", "probs_true", "probs_pred"]
    assert len(out["probs_true"]) == pred_models.shape[-1]
    assert len(out["probs_pred"]) == pred_models.shape[-1]
    # default: auto model names
    assert out["model_names"] == ["M_0", "M_1", "M_2"]

    # handles incorrect input?
    with pytest.raises(Exception):
        out = bf.diagnostics.metrics.expected_calibration_error(pred_models, true_models, model_names=["a"])

    with pytest.raises(Exception):
        out = bf.diagnostics.metrics.expected_calibration_error(pred_models, true_models.transpose)


def test_calibration_log_gamma(random_estimates, random_targets):
    out = bf.diagnostics.metrics.calibration_log_gamma(random_estimates, random_targets)
    assert list(out.keys()) == ["values", "metric_name", "variable_names"]
    assert out["values"].shape == (num_variables(random_estimates),)
    assert out["metric_name"] == "Log Gamma"
    assert out["variable_names"] == ["beta_0", "beta_1", "sigma"]


def test_calibration_log_gamma_end_to_end():
    # This is a function test for simulation-based calibration.
    # First, we sample from a known generative process and then run SBC.
    # If the log gamma statistic is correctly implemented, a 95% interval should exclude
    # the true value 5% of the time.

    N = 30  # number of samples
    S = 1000  # number of posterior draws
    D = 1000  # number of datasets

    def run_sbc(N=N, S=S, D=D, bias=0):
        rng = np.random.default_rng()
        prior_draws = rng.beta(2, 2, size=D)
        successes = rng.binomial(N, prior_draws)

        # Analytical posterior:
        # if theta ~ Beta(2, 2), then p(theta|successes) is Beta(2 + successes | 2 + N - successes).
        posterior_draws = rng.beta(2 + successes + bias, 2 + N - successes + bias, size=(S, D))

        # these ranks are uniform if bias=0
        ranks = np.sum(posterior_draws < prior_draws, axis=0)

        # this is the distribution of gamma under uniform ranks
        gamma_null = bf.diagnostics.metrics.gamma_null_distribution(D, S, num_null_draws=100)
        lower, upper = np.quantile(gamma_null, (0.05, 0.995))

        # this is the empirical gamma
        observed_gamma = bf.diagnostics.metrics.gamma_discrepancy(ranks, num_post_draws=S)

        in_interval = lower <= observed_gamma < upper

        return in_interval

    sbc_calibration = [run_sbc(N=N, S=S, D=D) for _ in range(100)]
    lower_expected, upper_expected = binom.ppf((0.0005, 0.9995), 100, 0.95)

    # this test should fail with a probability of 0.1%
    assert lower_expected <= np.sum(sbc_calibration) <= upper_expected

    # sbc should almost always fial for slightly biased posterior draws
    sbc_calibration = [run_sbc(N=N, S=S, D=D, bias=1) for _ in range(100)]
    assert not lower_expected <= np.sum(sbc_calibration) <= upper_expected


def test_bootstrap_comparison_shapes():
    """Test the bootstrap_comparison output shapes."""
    observed_samples = np.random.rand(10, 5)
    reference_samples = np.random.rand(100, 5)
    num_null_samples = 50

    distance_observed, distance_null = bf.diagnostics.metrics.bootstrap_comparison(
        observed_samples,
        reference_samples,
        lambda x, y: keras.ops.abs(keras.ops.mean(x) - keras.ops.mean(y)),
        num_null_samples,
    )

    assert isinstance(distance_observed, float)
    assert isinstance(distance_null, np.ndarray)
    assert distance_null.shape == (num_null_samples,)


def test_bootstrap_comparison_same_distribution():
    """Test bootstrap_comparison on same distributions."""
    observed_samples = np.random.normal(loc=0.5, scale=0.1, size=(10, 5))
    reference_samples = observed_samples.copy()
    num_null_samples = 5

    distance_observed, distance_null = bf.diagnostics.metrics.bootstrap_comparison(
        observed_samples,
        reference_samples,
        lambda x, y: keras.ops.abs(keras.ops.mean(x) - keras.ops.mean(y)),
        num_null_samples,
    )

    assert distance_observed <= np.quantile(distance_null, 0.99)


def test_bootstrap_comparison_different_distributions():
    """Test bootstrap_comparison on different distributions."""
    observed_samples = np.random.normal(loc=-5, scale=0.1, size=(10, 5))
    reference_samples = np.random.normal(loc=5, scale=0.1, size=(100, 5))
    num_null_samples = 50

    distance_observed, distance_null = bf.diagnostics.metrics.bootstrap_comparison(
        observed_samples,
        reference_samples,
        lambda x, y: keras.ops.abs(keras.ops.mean(x) - keras.ops.mean(y)),
        num_null_samples,
    )

    assert distance_observed >= np.quantile(distance_null, 0.68)


def test_bootstrap_comparison_mismatched_shapes():
    """Test bootstrap_comparison raises ValueError for mismatched shapes."""
    observed_samples = np.random.rand(10, 5)
    reference_samples = np.random.rand(20, 4)
    num_null_samples = 10

    with pytest.raises(ValueError):
        bf.diagnostics.metrics.bootstrap_comparison(
            observed_samples,
            reference_samples,
            lambda x, y: keras.ops.abs(keras.ops.mean(x) - keras.ops.mean(y)),
            num_null_samples,
        )


def test_bootstrap_comparison_num_observed_exceeds_num_reference():
    """Test bootstrap_comparison raises ValueError when number of observed samples exceeds the number of reference
    samples."""
    observed_samples = np.random.rand(100, 5)
    reference_samples = np.random.rand(20, 5)
    num_null_samples = 50

    with pytest.raises(ValueError):
        bf.diagnostics.metrics.bootstrap_comparison(
            observed_samples,
            reference_samples,
            lambda x, y: keras.ops.abs(keras.ops.mean(x) - keras.ops.mean(y)),
            num_null_samples,
        )


def test_mmd_comparison_from_summaries_shapes():
    """Test the mmd_comparison_from_summaries output shapes."""
    observed_summaries = np.random.rand(10, 5)
    reference_summaries = np.random.rand(100, 5)
    num_null_samples = 50

    mmd_observed, mmd_null = bf.diagnostics.metrics.bootstrap_comparison(
        observed_summaries,
        reference_summaries,
        comparison_fn=bf.metrics.functional.maximum_mean_discrepancy,
        num_null_samples=num_null_samples,
    )

    assert isinstance(mmd_observed, float)
    assert isinstance(mmd_null, np.ndarray)
    assert mmd_null.shape == (num_null_samples,)


def test_mmd_comparison_from_summaries_positive():
    """Test MMD output values of mmd_comparison_from_summaries are positive."""
    observed_summaries = np.random.rand(10, 5)
    reference_summaries = np.random.rand(100, 5)
    num_null_samples = 50

    mmd_observed, mmd_null = bf.diagnostics.metrics.bootstrap_comparison(
        observed_summaries,
        reference_summaries,
        comparison_fn=bf.metrics.functional.maximum_mean_discrepancy,
        num_null_samples=num_null_samples,
    )

    assert mmd_observed >= 0
    assert np.all(mmd_null >= 0)


def test_mmd_comparison_from_summaries_same_distribution():
    """Test mmd_comparison_from_summaries on same distributions."""
    observed_summaries = np.random.rand(10, 5)
    reference_summaries = observed_summaries.copy()
    num_null_samples = 5

    mmd_observed, mmd_null = bf.diagnostics.metrics.bootstrap_comparison(
        observed_summaries,
        reference_summaries,
        comparison_fn=bf.metrics.functional.maximum_mean_discrepancy,
        num_null_samples=num_null_samples,
    )

    assert mmd_observed <= np.quantile(mmd_null, 0.99)


def test_mmd_comparison_from_summaries_different_distributions():
    """Test mmd_comparison_from_summaries on different distributions."""
    observed_summaries = np.random.rand(10, 5)
    reference_summaries = np.random.normal(loc=0.5, scale=0.1, size=(100, 5))
    num_null_samples = 50

    mmd_observed, mmd_null = bf.diagnostics.metrics.bootstrap_comparison(
        observed_summaries,
        reference_summaries,
        comparison_fn=bf.metrics.functional.maximum_mean_discrepancy,
        num_null_samples=num_null_samples,
    )

    assert mmd_observed >= np.quantile(mmd_null, 0.68)


def test_mmd_comparison_shapes(summary_network, adapter):
    """Test the mmd_comparison output shapes."""
    observed_data = dict(observables=np.random.rand(10, 5))
    reference_data = dict(observables=np.random.rand(100, 5))
    num_null_samples = 50

    mock_approximator = bf.approximators.ContinuousApproximator(
        adapter=adapter,
        inference_network=None,
        summary_network=summary_network,
    )

    mmd_observed, mmd_null = bf.diagnostics.metrics.summary_space_comparison(
        observed_data=observed_data,
        reference_data=reference_data,
        approximator=mock_approximator,
        num_null_samples=num_null_samples,
        comparison_fn=bf.metrics.functional.maximum_mean_discrepancy,
    )

    assert isinstance(mmd_observed, float)
    assert isinstance(mmd_null, np.ndarray)
    assert mmd_null.shape == (num_null_samples,)


def test_mmd_comparison_positive(summary_network, adapter):
    """Test MMD output values of mmd_comparison are positive."""
    observed_data = dict(observables=np.random.rand(10, 5))
    reference_data = dict(observables=np.random.rand(100, 5))
    num_null_samples = 50

    mock_approximator = bf.approximators.ContinuousApproximator(
        adapter=adapter,
        inference_network=None,
        summary_network=summary_network,
    )

    mmd_observed, mmd_null = bf.diagnostics.metrics.summary_space_comparison(
        observed_data=observed_data,
        reference_data=reference_data,
        approximator=mock_approximator,
        num_null_samples=num_null_samples,
        comparison_fn=bf.metrics.functional.maximum_mean_discrepancy,
    )

    assert mmd_observed >= 0
    assert np.all(mmd_null >= 0)


def test_mmd_comparison_same_distribution(summary_network, adapter):
    """Test mmd_comparison on same distributions."""
    observed_data = dict(observables=np.random.rand(10, 5))
    reference_data = observed_data
    num_null_samples = 5

    mock_approximator = bf.approximators.ContinuousApproximator(
        adapter=adapter,
        inference_network=None,
        summary_network=summary_network,
    )

    mmd_observed, mmd_null = bf.diagnostics.metrics.summary_space_comparison(
        observed_data=observed_data,
        reference_data=reference_data,
        approximator=mock_approximator,
        num_null_samples=num_null_samples,
        comparison_fn=bf.metrics.functional.maximum_mean_discrepancy,
    )

    assert mmd_observed <= np.quantile(mmd_null, 0.99)


def test_mmd_comparison_different_distributions(summary_network, adapter):
    """Test mmd_comparison on different distributions."""
    observed_data = dict(observables=np.random.rand(10, 5))
    reference_data = dict(observables=np.random.normal(loc=0.5, scale=0.1, size=(100, 5)))
    num_null_samples = 50

    mock_approximator = bf.approximators.ContinuousApproximator(
        adapter=adapter,
        inference_network=None,
        summary_network=summary_network,
    )

    mmd_observed, mmd_null = bf.diagnostics.metrics.summary_space_comparison(
        observed_data=observed_data,
        reference_data=reference_data,
        approximator=mock_approximator,
        num_null_samples=num_null_samples,
        comparison_fn=bf.metrics.functional.maximum_mean_discrepancy,
    )

    assert mmd_observed >= np.quantile(mmd_null, 0.68)


def test_mmd_comparison_no_summary_network(adapter):
    observed_data = dict(observables=np.random.rand(10, 5))
    reference_data = dict(observables=np.random.rand(100, 5))
    num_null_samples = 50

    mock_approximator = bf.approximators.ContinuousApproximator(
        adapter=adapter,
        inference_network=None,
        summary_network=None,
    )

    with pytest.raises(ValueError):
        bf.diagnostics.metrics.summary_space_comparison(
            observed_data=observed_data,
            reference_data=reference_data,
            approximator=mock_approximator,
            num_null_samples=num_null_samples,
            comparison_fn=bf.metrics.functional.maximum_mean_discrepancy,
        )


def test_mmd_comparison_approximator_incorrect_instance():
    """Test mmd_comparison raises ValueError for incorrect approximator instance."""
    observed_data = dict(observables=np.random.rand(10, 5))
    reference_data = dict(observables=np.random.rand(100, 5))
    num_null_samples = 50

    class IncorrectApproximator:
        pass

    mock_approximator = IncorrectApproximator()

    with pytest.raises(ValueError):
        bf.diagnostics.metrics.summary_space_comparison(
            observed_data=observed_data,
            reference_data=reference_data,
            approximator=mock_approximator,
            num_null_samples=num_null_samples,
            comparison_fn=bf.metrics.functional.maximum_mean_discrepancy,
        )
