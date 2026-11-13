import keras
import numpy as np
import pytest
from scipy.special import expit
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

    # test without aggregation
    out = bf.diagnostics.metrics.calibration_error(random_estimates, random_targets, resolution=20, aggregation=None)
    assert out["values"].shape == (20, num_variables(random_estimates))

    # test quantities
    test_quantities = {
        r"$\beta_1 + \beta_2$": lambda data: np.sum(data["beta"], axis=-1),
        r"$\beta_1 \cdot \beta_2$": lambda data: np.prod(data["beta"], axis=-1),
    }
    out = bf.diagnostics.metrics.calibration_error(random_estimates, random_targets, test_quantities=test_quantities)
    assert out["values"].shape[0] == len(test_quantities) + num_variables(random_estimates)


def test_posterior_contraction(random_estimates, random_targets):
    # basic functionality: automatic variable names
    out = bf.diagnostics.metrics.posterior_contraction(random_estimates, random_targets)
    assert list(out.keys()) == ["values", "metric_name", "variable_names"]
    assert out["values"].shape == (num_variables(random_estimates),)
    assert out["metric_name"] == "Posterior Contraction"
    assert out["variable_names"] == ["beta_0", "beta_1", "sigma"]
    # test without aggregation
    out = bf.diagnostics.metrics.posterior_contraction(random_estimates, random_targets, aggregation=None)
    assert out["values"].shape == (random_estimates["sigma"].shape[0], num_variables(random_estimates))

    # test quantities
    test_quantities = {
        r"$\beta_1 + \beta_2$": lambda data: np.sum(data["beta"], axis=-1),
        r"$\beta_1 \cdot \beta_2$": lambda data: np.prod(data["beta"], axis=-1),
    }
    out = bf.diagnostics.metrics.posterior_contraction(
        random_estimates, random_targets, test_quantities=test_quantities
    )
    assert out["values"].shape[0] == len(test_quantities) + num_variables(random_estimates)


def test_posterior_z_score(random_estimates, random_targets):
    # basic functionality: automatic variable names
    out = bf.diagnostics.metrics.posterior_z_score(random_estimates, random_targets)
    assert list(out.keys()) == ["values", "metric_name", "variable_names"]
    assert out["values"].shape == (num_variables(random_estimates),)
    assert out["metric_name"] == "Posterior z-score"
    assert out["variable_names"] == ["beta_0", "beta_1", "sigma"]
    # test without aggregation
    out = bf.diagnostics.metrics.posterior_z_score(random_estimates, random_targets, aggregation=None)
    assert out["values"].shape == (random_estimates["sigma"].shape[0], num_variables(random_estimates))

    # test quantities
    test_quantities = {
        r"$\beta_1 + \beta_2$": lambda data: np.sum(data["beta"], axis=-1),
        r"$\beta_1 \cdot \beta_2$": lambda data: np.prod(data["beta"], axis=-1),
    }
    out = bf.diagnostics.metrics.posterior_z_score(random_estimates, random_targets, test_quantities=test_quantities)
    assert out["values"].shape[0] == len(test_quantities) + num_variables(random_estimates)


def test_root_mean_squared_error(random_estimates, random_targets):
    # basic functionality: automatic variable names
    out = bf.diagnostics.metrics.root_mean_squared_error(random_estimates, random_targets)
    assert list(out.keys()) == ["values", "metric_name", "variable_names"]
    assert out["values"].shape == (num_variables(random_estimates),)
    assert out["metric_name"] == "NRMSE"
    assert out["variable_names"] == ["beta_0", "beta_1", "sigma"]

    # test quantities
    test_quantities = {
        r"$\beta_1 + \beta_2$": lambda data: np.sum(data["beta"], axis=-1),
        r"$\beta_1 \cdot \beta_2$": lambda data: np.prod(data["beta"], axis=-1),
    }
    out = bf.diagnostics.metrics.root_mean_squared_error(
        random_estimates, random_targets, test_quantities=test_quantities
    )
    assert out["values"].shape[0] == len(test_quantities) + num_variables(random_estimates)


def test_canonical_correlation_metric_with_dict_inputs():
    summaries = {
        "summary": np.array(
            [
                [0.0, 1.0],
                [1.0, 0.0],
                [2.0, 1.0],
                [3.0, 2.0],
            ]
        )
    }
    sims = {"x": summaries["summary"][:, None, :]}

    out = bf.diagnostics.metrics.canonical_correlation_metric(
        summaries,
        sims,
        summary_keys="summary",
        target_keys="x",
    )

    assert list(out.keys()) == [
        "values",
        "metric_name",
        "variable_names",
    ]
    assert np.allclose(out["values"], [1.0, 1.0])
    assert out["metric_name"] == "Canonical Correlation Metric"
    assert out["variable_names"] == ["canonical_correlation_1", "canonical_correlation_2"]


def test_classifier_two_sample_test(random_samples_a, random_samples_b):
    c2st = bf.diagnostics.metrics.classifier_two_sample_test

    # a single number is returned, which is 0.5 for identical and larger for differing distributions
    metric = c2st(estimates=random_samples_a[:2000], targets=random_samples_a[2000:4000], cross_validation_splits=1)
    assert 0.55 > metric > 0.45

    metric = c2st(estimates=random_samples_a[:2000], targets=random_samples_b[:2000], cross_validation_splits=1)
    assert metric > 0.55

    # all statistics of all splits are returned; the conformal test is included automatically whenever
    # a dictionary is returned, and detects the difference
    metrics = c2st(
        estimates=random_samples_a[:2000],
        targets=random_samples_b[:2000],
        cross_validation_splits=3,
        return_metric_only=False,
    )
    assert len(metrics["scores"]) == len(metrics["conformal_p_values"]) == 3
    assert (np.array(metrics["scores"]) > 0.55).all()
    assert metrics["auc"] > 0.55
    assert metrics["regression_statistic"] > 0
    assert len(metrics["classifiers"]) == len(metrics["histories"]) == 3
    assert metrics["conformal_p_value"] < 0.01

    # unequal sample sizes: the classifier is trained with class weights and both the conformal test and
    # the rank-based statistics stay calibrated for samples from the same distribution
    metrics = c2st(
        estimates=random_samples_a[:3000],
        targets=random_samples_a[3000:4000],
        cross_validation_splits=1,
        conformal=True,
    )
    assert 0.4 < metrics["auc"] < 0.6
    assert metrics["conformal_p_value"] > 0.001
    assert "classifiers" not in metrics

    # the permutation p-values are resolved up to 1 / (num_permutations + 1). The two distributions differ
    # clearly, so the rank-based statistic attains the smallest possible p-value.
    num_permutations = 3
    metrics = c2st(
        estimates=random_samples_a[:1000],
        targets=random_samples_b[:1000],
        cross_validation_splits=1,
        max_epochs=20,
        num_permutations=num_permutations,
    )
    p_values = metrics["permutation_p_values"]
    assert set(p_values) == {"score", "regression_statistic", "auc"}
    assert all(1 / (num_permutations + 1) <= p <= 1 for p in p_values.values())
    assert p_values["auc"] == pytest.approx(1 / (num_permutations + 1))

    # a compiled model, an uncompiled model and a callable returning one are all accepted as classifiers
    def build_classifier():
        return keras.Sequential(
            [
                keras.layers.Dense(16, activation="relu"),
                keras.layers.Dense(1, activation="sigmoid"),
            ]
        )

    compiled = build_classifier()
    compiled.build((None, random_samples_a.shape[-1]))
    compiled.compile(
        optimizer=keras.optimizers.SGD(learning_rate=0.05), loss="binary_crossentropy", metrics=["accuracy"]
    )
    weights = [w.copy() for w in compiled.get_weights()]

    for classifier in (compiled, build_classifier(), build_classifier):
        metric = c2st(
            estimates=random_samples_a[:500],
            targets=random_samples_b[:500],
            classifier=classifier,
            cross_validation_splits=1,
            max_epochs=5,
        )
        assert 0 <= metric <= 1

    # the classifier that was passed in is cloned, not trained
    assert all(np.array_equal(before, after) for before, after in zip(weights, compiled.get_weights()))

    with pytest.raises(TypeError):
        c2st(estimates=random_samples_a, targets=random_samples_b, classifier="not-a-model", max_epochs=1)

    # the seed makes the splits, the conformal tie-breaking and the permutations reproducible
    kwargs = dict(
        estimates=random_samples_a[:400],
        targets=random_samples_b[:400],
        cross_validation_splits=1,
        max_epochs=5,
        conformal=True,
        num_permutations=2,
    )
    first, second, other = [c2st(**kwargs, seed=seed) for seed in (42, 42, 7)]
    keys = ("score", "auc", "regression_statistic", "conformal_p_value")
    assert all(first[key] == second[key] for key in keys)
    assert any(first[key] != other[key] for key in keys)


def test_classifier_two_sample_test_statistics():
    """Unit tests for the test statistics, using synthetic classifier scores."""
    from bayesflow.diagnostics.metrics.classifier_two_sample_test import (
        _make_splits,
        _predict_scores,
        area_under_the_curve,
        conformal_test,
        regression_statistic,
    )

    rng = np.random.default_rng(2026)

    # the AUC matches its definition, including the treatment of ties
    scores_a = rng.integers(0, 5, size=50).astype(float)
    scores_b = rng.integers(0, 5, size=70).astype(float)
    pairwise = (scores_a[:, None] > scores_b[None, :]) + 0.5 * (scores_a[:, None] == scores_b[None, :])
    assert area_under_the_curve(scores_a, scores_b) == pytest.approx(np.mean(pairwise))

    # the regression statistic is zero for an uninformative and maximal for a perfect classifier
    labels = np.r_[np.zeros(500), np.ones(500)]
    assert regression_statistic(np.full(1000, 0.5), labels) == pytest.approx(0.0)
    assert regression_statistic(labels, labels) == pytest.approx(0.25)

    # the ranking scores are the pre-sigmoid log-odds when the classifier ends in a dense sigmoid unit,
    # so they stay distinct where the float32 probabilities saturate to 1
    model = keras.Sequential([keras.layers.Dense(1, activation="sigmoid")])
    model.build((None, 1))
    model.layers[-1].set_weights([np.array([[20.0]]), np.array([0.0])])
    x = np.array([[1.0], [2.0], [3.0], [4.0]], dtype="float32")
    ranking_scores = _predict_scores(model, x)
    assert len(np.unique(np.asarray(model.predict(x, verbose=0)))) == 1
    assert np.allclose(ranking_scores, [20.0, 40.0, 60.0, 80.0])

    # away from saturation, the probabilities match the model output
    model.layers[-1].set_weights([np.array([[0.5]]), np.array([0.1])])
    ranking_scores = _predict_scores(model, x)
    probabilities = np.asarray(model.predict(x, verbose=0)).reshape(-1)
    assert np.allclose(expit(ranking_scores), probabilities, atol=1e-6)
    assert np.allclose(ranking_scores, 0.5 * x.reshape(-1) + 0.1)

    # models that do not end in a dense sigmoid unit fall back to the log-odds of their probabilities
    fallback = keras.Sequential([keras.layers.Dense(1), keras.layers.Activation("sigmoid")])
    fallback.build((None, 1))
    ranking_scores = _predict_scores(fallback, x)
    probabilities = np.asarray(fallback.predict(x, verbose=0)).reshape(-1)
    assert np.allclose(expit(ranking_scores), probabilities)

    # the splits are stratified, disjoint from the training data and cover all samples
    labels = np.r_[np.zeros(97), np.ones(103)]
    for splits in (_make_splits(labels, 1, 0.5, rng), _make_splits(labels, 4, 0.5, rng)):
        for train_idx, val_idx in splits:
            assert len(np.intersect1d(train_idx, val_idx)) == 0
            assert len(train_idx) + len(val_idx) == len(labels)
            assert labels[val_idx].mean() == pytest.approx(labels.mean(), abs=0.05)
    all_val = np.concatenate([val_idx for _, val_idx in _make_splits(labels, 4, 0.5, rng)])
    assert len(np.unique(all_val)) == len(labels)

    def rejection_rate(shift, num_reps):
        rejected = 0
        for _ in range(num_reps):
            reference = rng.normal(shift, 1, size=200)
            estimates = rng.normal(0, 1, size=100)
            rejected += conformal_test(reference, estimates, rng)[1] < 0.05
        return rejected / num_reps

    # the conformal test is calibrated if the scores of both samples follow the same distribution ...
    assert rejection_rate(0.0, 200) < 0.12
    # ... and powerful if they do not
    assert rejection_rate(1.0, 20) > 0.8


def test_model_comparison_accuracy():
    probs = np.array([[0.8, 0.1, 0.1], [0.1, 0.9, 0.0], [0.2, 0.3, 0.5]])
    one_hot = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

    # PMP mode: predictions shape (N, M)
    assert bf.diagnostics.metrics.accuracy(probs, one_hot, per_model=False) == 1.0

    # partial accuracy
    probs_wrong = np.array([[0.1, 0.8, 0.1], [0.1, 0.9, 0.0], [0.2, 0.3, 0.5]])
    acc = bf.diagnostics.metrics.accuracy(probs_wrong, one_hot, per_model=False)
    assert abs(acc - 2 / 3) < 1e-6


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
    assert out["model_names"] == ["M_1", "M_2", "M_3"]

    # handles incorrect input?
    with pytest.raises(Exception):
        out = bf.diagnostics.metrics.expected_calibration_error(pred_models, true_models, model_names=["a"])

    with pytest.raises(Exception):
        out = bf.diagnostics.metrics.expected_calibration_error(pred_models, true_models.transpose)


def test_model_comparison_brier_score(pred_models, true_models, model_names):
    out = bf.diagnostics.metrics.brier_score(pred_models, true_models, model_names=model_names)
    assert list(out.keys()) == ["values", "aggregate", "metric_name", "model_names"]
    assert out["values"].shape == (pred_models.shape[-1],)
    assert out["metric_name"] == "Brier Score"
    assert out["model_names"] == [r"$\mathcal{M}_0$", r"$\mathcal{M}_1$", r"$\mathcal{M}_2$"]
    assert abs(out["aggregate"] - out["values"].sum()) < 1e-6

    # default: auto model names
    out = bf.diagnostics.metrics.brier_score(pred_models, true_models)
    assert out["model_names"] == ["M_1", "M_2", "M_3"]

    # perfect predictions score 0, maximally wrong confident predictions score 2
    one_hot = np.eye(3)[np.array([0, 1, 2])]
    out = bf.diagnostics.metrics.brier_score(one_hot, one_hot)
    assert np.allclose(out["values"], 0.0)
    wrong = np.eye(3)[np.array([1, 2, 0])]
    out = bf.diagnostics.metrics.brier_score(wrong, one_hot)
    assert abs(out["aggregate"] - 2.0) < 1e-6


def test_calibration_log_gamma(random_estimates, random_targets):
    out = bf.diagnostics.metrics.calibration_log_gamma(random_estimates, random_targets)
    assert list(out.keys()) == ["values", "metric_name", "variable_names"]
    assert out["values"].shape == (num_variables(random_estimates),)
    assert out["metric_name"] == "Log Gamma"
    assert out["variable_names"] == ["beta_0", "beta_1", "sigma"]

    # test quantities
    test_quantities = {
        r"$\beta_1 + \beta_2$": lambda data: np.sum(data["beta"], axis=-1),
        r"$\beta_1 \cdot \beta_2$": lambda data: np.prod(data["beta"], axis=-1),
    }
    out = bf.diagnostics.metrics.calibration_log_gamma(
        random_estimates, random_targets, test_quantities=test_quantities
    )
    assert out["values"].shape[0] == len(test_quantities) + num_variables(random_estimates)


def test_calibration_log_gamma_end_to_end():
    # This is a function test for simulation-based calibration.
    # First, we sample from a known generative process and then run SBC.
    # If the log gamma statistic is correctly implemented, a 95% interval should exclude
    # the true value 5% of the time.

    N = 30  # number of samples
    S = 1000  # number of posterior draws
    D = 1000  # number of datasets

    gamma_null = bf.diagnostics.metrics.gamma_null_distribution(D, S, num_null_draws=10000)

    def run_sbc(N=N, S=S, D=D, bias=0):
        rng = np.random.default_rng()
        prior_draws = rng.beta(2, 2, size=D)
        successes = rng.binomial(N, prior_draws)

        # Analytical posterior:
        # if theta ~ Beta(2, 2), then p(theta|successes) is Beta(2 + successes | 2 + N - successes).
        posterior_draws = rng.beta(2 + successes + bias, 2 + N - successes, size=(S, D))

        # these ranks are uniform if bias=0
        ranks = np.sum(posterior_draws < prior_draws, axis=0)

        # this is the distribution of gamma under uniform ranks
        lower, upper = np.quantile(gamma_null, (0.025, 0.975))

        # this is the empirical gamma
        observed_gamma = bf.diagnostics.metrics.gamma_discrepancy(ranks, num_post_draws=S)

        in_interval = lower <= observed_gamma < upper

        return in_interval

    sbc_calibration = [run_sbc(N=N, S=S, D=D) for _ in range(100)]
    lower_expected, upper_expected = binom.ppf((0.00005, 0.99995), 100, 0.95)

    # this test should fail with a probability of 0.01%
    assert lower_expected <= np.sum(sbc_calibration) <= upper_expected

    # sbc should almost always fail for slightly biased posterior draws
    sbc_calibration = [run_sbc(N=N, S=S, D=D, bias=2) for _ in range(100)]
    assert not lower_expected <= np.sum(sbc_calibration) <= upper_expected


def test_accuracy_random_points():
    # overdispersed posterior should have a higher atc than underdispersed
    rng = np.random.default_rng(0)
    N, S, D = 500, 300, 3
    thetas = rng.normal(size=(N, D))
    wide = thetas[:, None, :] + rng.normal(scale=5.0, size=(N, S, D))
    narrow = thetas[:, None, :] + rng.normal(scale=0.01, size=(N, S, D))
    atc_wide = bf.diagnostics.metrics.accuracy_random_points(wide, thetas)["values"]
    atc_narrow = bf.diagnostics.metrics.accuracy_random_points(narrow, thetas)["values"]
    assert atc_wide > atc_narrow


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

    assert distance_observed.shape == ()
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

    assert mmd_observed.shape == ()
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

    mock_estimator = bf.estimators.BasicEstimator(
        adapter=adapter,
        inference_network=None,
        summary_network=summary_network,
    )

    mmd_observed, mmd_null = bf.diagnostics.metrics.summary_space_comparison(
        observed_data=observed_data,
        reference_data=reference_data,
        estimator=mock_estimator,
        num_null_samples=num_null_samples,
        comparison_fn=bf.metrics.functional.maximum_mean_discrepancy,
    )

    assert mmd_observed.shape == ()
    assert mmd_null.shape == (num_null_samples,)


def test_mmd_comparison_positive(summary_network, adapter):
    """Test MMD output values of mmd_comparison are positive."""
    observed_data = dict(observables=np.random.rand(10, 5))
    reference_data = dict(observables=np.random.rand(100, 5))
    num_null_samples = 50

    mock_estimator = bf.estimators.BasicEstimator(
        adapter=adapter,
        inference_network=None,
        summary_network=summary_network,
    )

    mmd_observed, mmd_null = bf.diagnostics.metrics.summary_space_comparison(
        observed_data=observed_data,
        reference_data=reference_data,
        estimator=mock_estimator,
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

    mock_estimator = bf.estimators.BasicEstimator(
        adapter=adapter,
        inference_network=None,
        summary_network=summary_network,
    )

    mmd_observed, mmd_null = bf.diagnostics.metrics.summary_space_comparison(
        observed_data=observed_data,
        reference_data=reference_data,
        estimator=mock_estimator,
        num_null_samples=num_null_samples,
        comparison_fn=bf.metrics.functional.maximum_mean_discrepancy,
    )

    assert mmd_observed <= np.quantile(mmd_null, 0.99)


def test_mmd_comparison_different_distributions(summary_network, adapter):
    """Test mmd_comparison on different distributions."""
    observed_data = dict(observables=np.random.rand(10, 5))
    reference_data = dict(observables=np.random.normal(loc=0.5, scale=0.1, size=(100, 5)))
    num_null_samples = 50

    mock_estimator = bf.estimators.BasicEstimator(
        adapter=adapter,
        inference_network=None,
        summary_network=summary_network,
    )

    mmd_observed, mmd_null = bf.diagnostics.metrics.summary_space_comparison(
        observed_data=observed_data,
        reference_data=reference_data,
        estimator=mock_estimator,
        num_null_samples=num_null_samples,
        comparison_fn=bf.metrics.functional.maximum_mean_discrepancy,
    )

    assert mmd_observed >= np.quantile(mmd_null, 0.68)


def test_mmd_comparison_no_summary_network(adapter):
    observed_data = dict(observables=np.random.rand(10, 5))
    reference_data = dict(observables=np.random.rand(100, 5))
    num_null_samples = 50

    mock_estimator = bf.estimators.BasicEstimator(
        adapter=adapter,
        inference_network=None,
        summary_network=None,
    )

    with pytest.raises(ValueError):
        bf.diagnostics.metrics.summary_space_comparison(
            observed_data=observed_data,
            reference_data=reference_data,
            estimator=mock_estimator,
            num_null_samples=num_null_samples,
            comparison_fn=bf.metrics.functional.maximum_mean_discrepancy,
        )


def test_mmd_comparison_estimator_incorrect_instance():
    """Test mmd_comparison raises ValueError for incorrect estimator instance."""
    observed_data = dict(observables=np.random.rand(10, 5))
    reference_data = dict(observables=np.random.rand(100, 5))
    num_null_samples = 50

    class IncorrectEstimator:
        pass

    mock_estimator = IncorrectEstimator()

    with pytest.raises(ValueError):
        bf.diagnostics.metrics.summary_space_comparison(
            observed_data=observed_data,
            reference_data=reference_data,
            estimator=mock_estimator,
            num_null_samples=num_null_samples,
            comparison_fn=bf.metrics.functional.maximum_mean_discrepancy,
        )


def _gaussian(n, dim=3, loc=0.0, seed=0):
    # fixed covariance, so draws with different seeds come from the same distribution
    a = np.random.default_rng(12345).normal(size=(dim, dim))
    cov = a @ a.T + dim * np.eye(dim)
    rng = np.random.default_rng(seed)
    return rng.multivariate_normal(mean=np.full(dim, loc), cov=cov, size=n)


def test_mahalanobis_distance_matches_scipy():
    from scipy.spatial.distance import mahalanobis

    reference = _gaussian(500, seed=1)
    samples = _gaussian(20, seed=2)

    distances = bf.diagnostics.metrics.mahalanobis_distance(samples, reference, regularization=0.0)
    squared = bf.diagnostics.metrics.mahalanobis_distance(samples, reference, regularization=0.0, squared=True)

    mean = reference.mean(axis=0)
    precision = np.linalg.inv(np.cov(reference, rowvar=False))
    expected = np.array([mahalanobis(s, mean, precision) for s in samples])

    assert distances.shape == (20,)
    np.testing.assert_allclose(distances, expected, rtol=1e-6)
    np.testing.assert_allclose(squared, expected**2, rtol=1e-6)


def test_mahalanobis_distance_flattens_trailing_dims():
    reference = np.random.normal(size=(200, 2, 3))
    samples = np.random.normal(size=(7, 2, 3))

    distances = bf.diagnostics.metrics.mahalanobis_distance(samples, reference)
    flat = bf.diagnostics.metrics.mahalanobis_distance(samples.reshape(7, 6), reference.reshape(200, 6))

    np.testing.assert_allclose(distances, flat)


def test_mahalanobis_distance_rank_deficient_covariance():
    reference = np.random.normal(size=(100, 2))
    # third column is a copy of the first, so the covariance is singular
    reference = np.concatenate([reference, reference[:, :1]], axis=-1)
    samples = np.random.normal(size=(10, 3))

    distances = bf.diagnostics.metrics.mahalanobis_distance(samples, reference)

    assert np.all(np.isfinite(distances))
    assert np.all(distances >= 0)


def test_mahalanobis_distance_invariant_to_rescaling():
    # the default ridge must not let large-variance dimensions swamp small-variance ones
    reference = _gaussian(5000, dim=2)
    samples = _gaussian(50, dim=2, seed=1)
    scales = np.array([1e3, 1e-3])

    distances = bf.diagnostics.metrics.mahalanobis_distance(samples, reference)
    rescaled = bf.diagnostics.metrics.mahalanobis_distance(samples * scales, reference * scales)

    np.testing.assert_allclose(rescaled, distances, rtol=1e-4)


def test_mahalanobis_distance_invalid_inputs():
    reference = np.random.normal(size=(100, 3))
    samples = np.random.normal(size=(10, 3))

    with pytest.raises(ValueError):
        bf.diagnostics.metrics.mahalanobis_distance(samples, reference[:1])

    with pytest.raises(ValueError):
        bf.diagnostics.metrics.mahalanobis_distance(np.where(samples > 1, np.nan, samples), reference)

    with pytest.raises(ValueError):
        bf.diagnostics.metrics.mahalanobis_distance(samples, reference, regularization=-1.0)

    with pytest.warns(UserWarning, match="reference samples"):
        bf.diagnostics.metrics.mahalanobis_distance(np.random.normal(size=(10, 50)), np.random.normal(size=(30, 50)))


def test_mahalanobis_distance_mismatched_shapes():
    with pytest.raises(ValueError):
        bf.diagnostics.metrics.mahalanobis_distance(np.random.rand(10, 5), np.random.rand(100, 4))


def test_mahalanobis_test_false_positive_rate():
    alpha = 0.05
    num_observed = 2000
    reference = _gaussian(20_000, seed=3)
    calibration = _gaussian(2000, seed=4)
    observed = _gaussian(num_observed, seed=5)

    result = bf.diagnostics.metrics.mahalanobis_test(observed, reference, calibration, alpha=alpha)

    # observed data come from the reference distribution, so the rejection rate must be close to alpha;
    # estimating the quantile on the calibration samples adds variance
    variance = alpha * (1 - alpha) * (1 / num_observed + 1 / len(calibration))
    assert abs(result["is_ood"].mean() - alpha) <= 4 * np.sqrt(variance)


@pytest.mark.parametrize("num_reference, dim", [(100, 8), (200, 32)])
def test_mahalanobis_test_false_positive_rate_small_reference(num_reference, dim):
    # mean and covariance are estimated from few samples, which inflates distances; the calibration samples
    # see the same estimation error, so the test stays calibrated (a chi-square threshold would not)
    alpha = 0.05
    rejection_rates = []
    for seed in range(20):
        reference = _gaussian(num_reference, dim=dim, seed=3 * seed)
        calibration = _gaussian(500, dim=dim, seed=3 * seed + 1)
        observed = _gaussian(500, dim=dim, seed=3 * seed + 2)
        result = bf.diagnostics.metrics.mahalanobis_test(observed, reference, calibration, alpha=alpha)
        rejection_rates.append(result["is_ood"].mean())

    # averaged over draws of the reference and calibration samples, the conformal test has level alpha
    assert abs(np.mean(rejection_rates) - alpha) <= 0.015


@pytest.mark.parametrize("squared", [False, True])
def test_mahalanobis_test_threshold_consistent_with_p_values(squared):
    reference = _gaussian(1000)
    calibration = _gaussian(99, seed=1)
    observed = _gaussian(500, seed=2)

    result = bf.diagnostics.metrics.mahalanobis_test(observed, reference, calibration, alpha=0.05, squared=squared)

    np.testing.assert_array_equal(result["is_ood"], result["p_values"] <= 0.05)
    np.testing.assert_array_equal(result["is_ood"], result["distances"] > result["threshold"])
    assert np.all((result["p_values"] > 0) & (result["p_values"] <= 1))


@pytest.mark.parametrize("alpha", [0.01, 0.05, 1 / 3, 0.7, 0.95, 0.99])
@pytest.mark.parametrize("num_calibration", [9, 10, 99, 1000])
def test_mahalanobis_test_threshold_round_off(alpha, num_calibration):
    # (n + 1) * (1 - alpha) can land just above an integer, which must not shift the reported threshold
    reference = _gaussian(500)
    calibration = _gaussian(num_calibration, seed=1)
    observed = _gaussian(2000, seed=2)

    result = bf.diagnostics.metrics.mahalanobis_test(observed, reference, calibration, alpha=alpha)

    np.testing.assert_array_equal(result["is_ood"], result["p_values"] <= alpha)
    np.testing.assert_array_equal(result["is_ood"], result["distances"] > result["threshold"])


def test_mahalanobis_test_squared():
    # squaring changes only the scale of distances and threshold, not p-values or flags
    reference = _gaussian(1000)
    calibration = _gaussian(200, seed=1)
    observed = _gaussian(100, loc=0.3, seed=2)

    plain = bf.diagnostics.metrics.mahalanobis_test(observed, reference, calibration)
    squared = bf.diagnostics.metrics.mahalanobis_test(observed, reference, calibration, squared=True)

    np.testing.assert_allclose(squared["distances"], plain["distances"] ** 2)
    np.testing.assert_allclose(squared["threshold"], plain["threshold"] ** 2)
    np.testing.assert_array_equal(squared["p_values"], plain["p_values"])
    np.testing.assert_array_equal(squared["is_ood"], plain["is_ood"])


def test_mahalanobis_test_too_few_calibration_samples():
    # with n calibration samples, the smallest p-value is 1 / (n + 1), so nothing can be flagged at alpha < 1 / 20
    reference = _gaussian(500)
    observed = _gaussian(10, loc=50.0, seed=1)

    result = bf.diagnostics.metrics.mahalanobis_test(observed, reference, _gaussian(18, seed=2), alpha=0.05)

    assert result["threshold"] == np.inf
    assert not result["is_ood"].any()


@pytest.mark.parametrize("degenerate", ["constant", "constant_nonzero", "collinear"])
def test_mahalanobis_test_rank_deficient(degenerate):
    # constant or collinear dimensions must neither break the test nor change the distances beyond the ridge
    alpha = 0.05
    reference = _gaussian(20_000, dim=2, seed=3)
    calibration = _gaussian(2000, dim=2, seed=4)
    observed = _gaussian(4000, dim=2, seed=5)

    def extend(x):
        match degenerate:
            case "constant":
                extra = np.zeros((len(x), 1))
            case "constant_nonzero":
                # np.cov returns a tiny round-off variance for this column
                extra = np.full((len(x), 1), 0.1)
            case "collinear":
                extra = x[:, :1] + 2 * x[:, 1:]
        return np.concatenate([x, extra], axis=-1)

    result = bf.diagnostics.metrics.mahalanobis_test(
        extend(observed), extend(reference), extend(calibration), alpha=alpha
    )

    variance = alpha * (1 - alpha) * (1 / len(observed) + 1 / len(calibration))
    assert abs(result["is_ood"].mean() - alpha) <= 4 * np.sqrt(variance)
    expected = bf.diagnostics.metrics.mahalanobis_distance(observed, reference)
    np.testing.assert_allclose(result["distances"], expected, rtol=1e-4)


def test_mahalanobis_empty_and_singular_inputs():
    reference = _gaussian(100)

    assert bf.diagnostics.metrics.mahalanobis_distance(reference[:0], reference).shape == (0,)

    with pytest.raises(ValueError):
        bf.diagnostics.metrics.mahalanobis_test(reference[:10], reference, reference[:0])

    singular = np.concatenate([reference, np.zeros((100, 1))], axis=-1)
    with pytest.raises(ValueError, match="singular"):
        bf.diagnostics.metrics.mahalanobis_distance(singular[:10], singular, regularization=0.0)


def test_mahalanobis_test_warnings_point_to_caller():
    reference = np.random.normal(size=(30, 50))

    with pytest.warns(UserWarning, match="reference samples") as record:
        bf.diagnostics.metrics.mahalanobis_test(reference[:10], reference, np.random.normal(size=(20, 50)))

    assert record[0].filename == __file__


def test_mahalanobis_test_constant_reference():
    reference = np.ones((100, 3))
    observed = np.r_[np.ones((5, 3)), np.full((5, 3), 2.0)]

    # any deviation from the constant is flagged
    result = bf.diagnostics.metrics.mahalanobis_test(observed, reference, np.ones((99, 3)))
    np.testing.assert_array_equal(result["is_ood"], [False] * 5 + [True] * 5)


def test_mahalanobis_test_mismatched_shapes():
    with pytest.raises(ValueError, match="observed_samples"):
        bf.diagnostics.metrics.mahalanobis_test(np.random.rand(10, 5), np.random.rand(100, 4), np.random.rand(50, 4))

    with pytest.raises(ValueError, match="calibration_samples"):
        bf.diagnostics.metrics.mahalanobis_test(np.random.rand(10, 4), np.random.rand(100, 4), np.random.rand(50, 5))


def test_mahalanobis_test_input_hints():
    reference = _gaussian(100)
    calibration = _gaussian(100, seed=1)

    # a single sample without a leading axis
    with pytest.raises(ValueError, match=r"observed_samples\[None\]"):
        bf.diagnostics.metrics.mahalanobis_test(reference[0], reference, calibration)

    # data dicts instead of summaries
    with pytest.raises(ValueError, match="summary_space_mahalanobis"):
        bf.diagnostics.metrics.mahalanobis_test({"x": reference[:10]}, reference, calibration)

    with pytest.raises(ValueError, match="scalar"):
        bf.diagnostics.metrics.mahalanobis_test(np.float64(1.0), reference, calibration)

    # trailing dimensions only need to match after flattening
    result = bf.diagnostics.metrics.mahalanobis_test(reference[:10].reshape(10, 3, 1), reference, calibration)
    assert result["distances"].shape == (10,)


def test_summary_space_mahalanobis_incorrect_estimator_instance():
    class IncorrectEstimator:
        pass

    with pytest.raises(ValueError):
        bf.diagnostics.metrics.summary_space_mahalanobis(
            observed_data=dict(observables=np.random.rand(10, 5)),
            reference_data=dict(observables=np.random.rand(100, 5)),
            estimator=IncorrectEstimator(),
            calibration_data=dict(observables=np.random.rand(100, 5)),
        )

    # calibration data passed positionally in place of the estimator
    with pytest.raises(ValueError, match="calibration_data="):
        bf.diagnostics.metrics.summary_space_mahalanobis(
            dict(observables=np.random.rand(10, 5)),
            dict(observables=np.random.rand(100, 5)),
            dict(observables=np.random.rand(100, 5)),
            calibration_data=dict(observables=np.random.rand(100, 5)),
        )


def test_mahalanobis_test_detects_shift():
    reference = _gaussian(5000)
    calibration = _gaussian(1000, seed=2)
    observed = _gaussian(200, loc=15.0, seed=1)

    result = bf.diagnostics.metrics.mahalanobis_test(observed, reference, calibration)

    assert result["is_ood"].all()


def test_mahalanobis_test_invalid_arguments():
    reference = _gaussian(100)
    calibration = _gaussian(100, seed=2)
    observed = _gaussian(10, seed=1)

    for alpha in [0.0, 1.0, 1.5]:
        with pytest.raises(ValueError):
            bf.diagnostics.metrics.mahalanobis_test(observed, reference, calibration, alpha=alpha)


def test_summary_space_mahalanobis(summary_network, adapter):
    observed_data = dict(observables=np.random.normal(loc=5.0, size=(10, 5)))
    reference_data = dict(observables=np.random.normal(size=(500, 5)))
    calibration_data = dict(observables=np.random.normal(size=(200, 5)))

    mock_estimator = bf.estimators.BasicEstimator(
        adapter=adapter,
        inference_network=None,
        summary_network=summary_network,
    )

    result = bf.diagnostics.metrics.summary_space_mahalanobis(
        observed_data=observed_data,
        reference_data=reference_data,
        estimator=mock_estimator,
        calibration_data=calibration_data,
    )

    assert set(result) == {"distances", "p_values", "threshold", "is_ood"}
    assert result["distances"].shape == (10,)
    assert result["is_ood"].all()


def test_summary_space_mahalanobis_no_summary_network(adapter):
    mock_estimator = bf.estimators.BasicEstimator(
        adapter=adapter,
        inference_network=None,
        summary_network=None,
    )

    with pytest.raises(ValueError):
        bf.diagnostics.metrics.summary_space_mahalanobis(
            observed_data=dict(observables=np.random.rand(10, 5)),
            reference_data=dict(observables=np.random.rand(100, 5)),
            estimator=mock_estimator,
            calibration_data=dict(observables=np.random.rand(100, 5)),
        )


@pytest.mark.slow
def test_summary_space_mahalanobis_trained_summary_network():
    """End-to-end: with a plainly trained summary network, the test is calibrated and detects shifts."""

    def prior():
        return dict(theta=np.random.normal(size=2))

    def likelihood(theta):
        return dict(x=np.random.normal(loc=theta, size=(20, 2)))

    simulator = bf.make_simulator([prior, likelihood])
    adapter = (
        bf.Adapter()
        .convert_dtype("float64", "float32")
        .rename("theta", "inference_variables")
        .rename("x", "summary_variables")
    )
    workflow = bf.BasicWorkflow(
        simulator=simulator,
        adapter=adapter,
        inference_network=bf.networks.CouplingFlow(depth=2),
        summary_network=bf.networks.DeepSet(summary_dim=4),
    )
    workflow.fit_online(epochs=10, num_batches_per_epoch=50, batch_size=64, verbose=0)

    reference = {"x": simulator.sample(5000)["x"]}
    calibration = {"x": simulator.sample(2000)["x"]}
    observed = {"x": simulator.sample(2000)["x"]}

    result = bf.diagnostics.metrics.summary_space_mahalanobis(
        observed, reference, workflow.estimator, calibration_data=calibration
    )
    shifted = bf.diagnostics.metrics.summary_space_mahalanobis(
        {"x": observed["x"] + 10.0}, reference, workflow.estimator, calibration_data=calibration
    )

    assert abs(result["is_ood"].mean() - 0.05) <= 4 * np.sqrt(0.05 * 0.95 * (1 / 2000 + 1 / 2000))
    assert shifted["is_ood"].mean() >= 0.9
