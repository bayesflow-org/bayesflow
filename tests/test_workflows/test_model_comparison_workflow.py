import os

import keras
import numpy as np
import pandas as pd
import pytest

from bayesflow.workflows import ModelComparisonWorkflow
from tests.utils import assert_models_equal


def test_pmp_workflow(tmp_path, mc_simulators):
    """End-to-end test with the default CrossEntropyScore (PMP mode)."""
    workflow = ModelComparisonWorkflow(
        simulator=mc_simulators,
        inference_conditions=["x"],
        checkpoint_filepath=str(tmp_path),
    )

    history = workflow.fit_online(epochs=2, batch_size=4, num_batches_per_epoch=2, verbose=0)
    plots = workflow.plot_default_diagnostics(test_data=20)
    metrics = workflow.compute_default_diagnostics(test_data=20)

    assert "loss" in history.history
    assert len(history.history["loss"]) == 2

    # Diagnostics: combined confusion + pairwise-BF figure + calibration (+ loss curve)
    assert set(plots.keys()) == {"loss", "confusion_and_bayes_factors", "calibration"}

    # Raw metric dicts (as_data_frame=False): per-model accuracy, ECE, and Brier score
    raw_metrics = workflow.compute_default_diagnostics(test_data=20, as_data_frame=False)
    assert set(raw_metrics) == {"accuracy", "ece", "brier_score"}
    assert raw_metrics["accuracy"]["values"].shape == (2,)
    assert raw_metrics["brier_score"]["values"].shape == (2,)
    assert 0.0 <= raw_metrics["brier_score"]["aggregate"] <= 2.0

    # PMP metrics (default): a DataFrame with one row per metric, one column per model
    assert isinstance(metrics, pd.DataFrame)
    assert set(metrics.index) == {"Accuracy", "Expected Calibration Error", "Brier Score"}
    assert list(metrics.columns) == list(raw_metrics["brier_score"]["model_names"])
    assert metrics.shape == (3, 2)
    assert ((metrics.loc["Accuracy"] >= 0.0) & (metrics.loc["Accuracy"] <= 1.0)).all()

    # Save/load round-trip
    loaded = keras.saving.load_model(os.path.join(str(tmp_path), "model.keras"))
    assert_models_equal(workflow.approximator, loaded)


def test_pmp_workflow_with_summary_network(mc_simulators, mc_summary_network):
    """PMP workflow with a summary network compresses observations before classifying."""
    workflow = ModelComparisonWorkflow(
        simulator=mc_simulators,
        summary_network=mc_summary_network,
        summary_variables=["x"],
    )

    workflow.fit_online(epochs=2, batch_size=4, num_batches_per_epoch=2, verbose=0)
    plots = workflow.plot_default_diagnostics(test_data=20)

    assert "confusion_and_bayes_factors" in plots
    assert "calibration" in plots


def test_bf_workflow(tmp_path, mc_simulators):
    """End-to-end test with ExponentialScore (Bayes factor mode)."""
    from bayesflow.scoring_rules import ExponentialScore

    workflow = ModelComparisonWorkflow(
        simulator=mc_simulators,
        scoring_rules=ExponentialScore(),
        inference_conditions=["x"],
        checkpoint_filepath=str(tmp_path),
    )

    history = workflow.fit_online(epochs=2, batch_size=4, num_batches_per_epoch=2, verbose=0)
    plots = workflow.plot_default_diagnostics(test_data=20)
    metrics = workflow.compute_default_diagnostics(test_data=20)

    assert "loss" in history.history
    assert len(history.history["loss"]) == 2

    # Diagnostics: combined confusion + pairwise-BF figure + calibration (+ loss)
    assert "loss" in plots
    assert "calibration" in plots
    assert "confusion_and_bayes_factors" in plots
    assert "bayes_factor_recovery" not in plots

    # BF metrics (default DataFrame): accuracy + ECE and Brier score from the derived model_probs
    assert isinstance(metrics, pd.DataFrame)
    assert set(metrics.index) == {"Accuracy", "Expected Calibration Error", "Brier Score"}
    assert metrics.shape[1] == 2
    assert ((metrics.loc["Accuracy"] >= 0.0) & (metrics.loc["Accuracy"] <= 1.0)).all()

    # Save/load round-trip
    loaded = keras.saving.load_model(os.path.join(str(tmp_path), "model.keras"))
    assert_models_equal(workflow.approximator, loaded)


def test_bf_workflow_with_bayes_factor_recovery(mc_simulators):
    """Supplying true_log_bfs_fn adds a bayes_factor_recovery plot for BF rules."""
    from bayesflow.scoring_rules import ExponentialScore

    workflow = ModelComparisonWorkflow(
        simulator=mc_simulators,
        scoring_rules=ExponentialScore(),
        inference_conditions=["x"],
    )
    workflow.fit_online(epochs=2, batch_size=4, num_batches_per_epoch=2, verbose=0)

    def true_log_bfs_fn(data):
        # Placeholder ground truth: zeros (same shape as predicted log BFs)
        return np.zeros((data["model_indices"].shape[0], 1))

    plots = workflow.plot_default_diagnostics(test_data=20, true_log_bfs_fn=true_log_bfs_fn)

    assert "bayes_factor_recovery" in plots
    assert "calibration" in plots
    assert "confusion_and_bayes_factors" in plots


def test_bf_workflow_with_summary_network(mc_simulators, mc_summary_network):
    """BF workflow with a summary network."""
    from bayesflow.links import Leaky
    from bayesflow.scoring_rules import ExponentialScore

    workflow = ModelComparisonWorkflow(
        simulator=mc_simulators,
        scoring_rules=ExponentialScore(links={"logits": Leaky(power=2.0)}),
        summary_network=mc_summary_network,
        summary_variables=["x"],
    )
    workflow.fit_online(epochs=2, batch_size=4, num_batches_per_epoch=2, verbose=0)
    plots = workflow.plot_default_diagnostics(test_data=20)

    assert "calibration" in plots
    assert "confusion_and_bayes_factors" in plots


@pytest.mark.parametrize("rule_name", ["cross_entropy", "brier", "exponential"])
def test_estimate_shapes(mc_simulators, rule_name):
    """estimate returns model_probs and log_odds; Bayes factors require an explicit prior."""
    num_models = len(mc_simulators)
    n_test = 10

    workflow = ModelComparisonWorkflow(
        simulator=mc_simulators,
        scoring_rules=rule_name,
        inference_conditions=["x"],
    )
    workflow.fit_online(epochs=2, batch_size=4, num_batches_per_epoch=2, verbose=0)
    test_data = workflow.simulate(n_test)

    estimates = workflow.estimate(conditions=test_data)

    assert isinstance(estimates, dict)
    assert estimates["model_probs"].shape == (n_test, num_models)
    assert estimates["log_odds"].shape == (n_test, num_models)
    assert "log_bayes_factors" not in estimates
    assert np.allclose(estimates["model_probs"].sum(axis=-1), 1.0, atol=1e-5)
    assert np.allclose(estimates["log_odds"][:, 0], 0.0, atol=1e-6)


def test_estimate_log_bayes_factors_with_model_prior(mc_simulators):
    """estimate adds log_bayes_factors when a model prior is provided."""
    num_models = len(mc_simulators)
    n_test = 8
    model_prior = np.linspace(1.0, 2.0, num_models)
    model_prior = model_prior / model_prior.sum()

    workflow = ModelComparisonWorkflow(
        simulator=mc_simulators,
        scoring_rules=["cross_entropy", "brier"],
        inference_conditions=["x"],
    )
    workflow.fit_online(epochs=1, batch_size=4, num_batches_per_epoch=2, verbose=0)
    test_data = workflow.simulate(n_test)

    nested = workflow.estimate(conditions=test_data, merge_scores=False, model_prior=model_prior)
    log_prior_odds = np.log(model_prior) - np.log(model_prior[0])

    for rule_result in nested.values():
        assert set(rule_result) == {"log_odds", "model_probs", "log_bayes_factors"}
        assert rule_result["log_bayes_factors"].shape == (n_test, num_models)
        assert np.allclose(rule_result["log_bayes_factors"], rule_result["log_odds"] - log_prior_odds, atol=1e-6)

    merged = workflow.estimate(conditions=test_data, model_prior=model_prior)
    assert set(merged) == {"log_odds", "model_probs", "log_bayes_factors"}
    assert np.allclose(merged["log_bayes_factors"], merged["log_odds"] - log_prior_odds, atol=1e-6)


def test_resolve_scoring_rules_normalizes_inputs():
    """_resolve_scoring_rules accepts single/str/list/dict and auto-names list entries."""
    from bayesflow.scoring_rules import CrossEntropyScore, BrierScore, ExponentialScore

    resolve = ModelComparisonWorkflow._resolve_scoring_rules

    # string name -> single rule
    assert list(resolve("exponential")) == ["scoring_rule"]

    # list -> snake-cased class-name keys
    assert list(resolve([CrossEntropyScore(), BrierScore()])) == ["cross_entropy_score", "brier_score"]

    # duplicate class -> numeric suffix
    assert list(resolve([ExponentialScore(), ExponentialScore()])) == ["exponential_score", "exponential_score_2"]

    # dict with a string value gets resolved to an instance
    assert resolve({"a": "brier"})["a"]


def test_mixed_pmp_and_bf_rules_workflow(mc_simulators):
    """PMP and BF rules co-train on separate heads and merge in log-odds space."""
    from bayesflow.scoring_rules import CrossEntropyScore, LogisticScore

    num_models = len(mc_simulators)
    n_test = 8

    workflow = ModelComparisonWorkflow(
        simulator=mc_simulators,
        scoring_rules=[CrossEntropyScore(), LogisticScore()],
        inference_conditions=["x"],
    )
    workflow.fit_online(epochs=1, batch_size=4, num_batches_per_epoch=2, verbose=0)
    test_data = workflow.simulate(n_test)

    nested = workflow.estimate(conditions=test_data, merge_scores=False)
    assert set(nested) == {"cross_entropy_score", "logistic_score"}
    assert set(nested["cross_entropy_score"]) == {"log_odds", "model_probs"}
    assert set(nested["logistic_score"]) == {"log_odds", "model_probs"}

    # Merged: logarithmic opinion pool over the length-M log_odds.
    merged = workflow.estimate(conditions=test_data)
    assert set(merged) == {"log_odds", "model_probs"}
    assert merged["log_odds"].shape == (n_test, num_models)
    expected = np.mean([nested["cross_entropy_score"]["log_odds"], nested["logistic_score"]["log_odds"]], axis=0)
    assert np.allclose(merged["log_odds"], expected, atol=1e-5)
    assert np.allclose(merged["model_probs"].sum(-1), 1.0, atol=1e-5)

    # Diagnostics: with both families present, the confusion matrix and pairwise
    # Bayes factor heatmap share a single side-by-side figure; plus all scalar metrics.
    plots = workflow.plot_default_diagnostics(test_data=test_data)
    assert {"confusion_and_bayes_factors", "calibration"} <= set(plots)
    assert "confusion_matrix" not in plots and "pairwise_bayes_factors" not in plots

    metrics = workflow.compute_default_diagnostics(test_data=test_data)
    assert isinstance(metrics, pd.DataFrame)
    assert set(metrics.index) == {"Accuracy", "Expected Calibration Error", "Brier Score"}
    # Same scalar metrics regardless of family; as_data_frame=False exposes the raw dicts.
    raw_metrics = workflow.compute_default_diagnostics(test_data=test_data, as_data_frame=False)
    assert {"accuracy", "ece", "brier_score"} <= set(raw_metrics)


@pytest.mark.parametrize("rule_names", [["cross_entropy", "brier"], ["exponential", "logistic"]])
def test_estimate_merge_scores(mc_simulators, rule_names):
    """merge_scores=False keeps per-rule nesting; True log-pools log_odds into a consistent flat dict."""
    num_models = len(mc_simulators)
    n_test = 8

    workflow = ModelComparisonWorkflow(
        simulator=mc_simulators,
        scoring_rules=rule_names,
        inference_conditions=["x"],
    )
    workflow.fit_online(epochs=1, batch_size=4, num_batches_per_epoch=2, verbose=0)
    test_data = workflow.simulate(n_test)

    # merge_scores=False -> nested dict keyed by rule name, each with the single-rule structure
    nested = workflow.estimate(conditions=test_data, merge_scores=False)
    assert len(nested) == 2
    for rule_result in nested.values():
        assert set(rule_result) == {"log_odds", "model_probs"}
        assert rule_result["model_probs"].shape == (n_test, num_models)

    # merge_scores=True (default) -> flat, logarithmic opinion pool over log_odds
    merged = workflow.estimate(conditions=test_data)
    assert set(merged) == {"log_odds", "model_probs"}
    assert merged["log_odds"].shape == (n_test, num_models)
    expected = np.mean([nested[k]["log_odds"] for k in nested], axis=0)
    assert np.allclose(merged["log_odds"], expected, atol=1e-5)

    # model_probs is derived from the pooled log_odds via softmax -> mutually consistent
    implied = np.exp(merged["log_odds"] - merged["log_odds"].max(-1, keepdims=True))
    implied /= implied.sum(-1, keepdims=True)
    assert np.allclose(merged["model_probs"], implied, atol=1e-5)
    assert np.allclose(merged["model_probs"].sum(-1), 1.0, atol=1e-5)


def test_requires_at_least_two_simulators():
    """Fewer than 2 simulators raises ValueError."""
    from bayesflow import make_simulator

    def prior():
        return dict(mu=0.0)

    def likelihood(mu):
        return dict(x=np.random.normal(mu, 1, 4).astype(np.float32))

    with pytest.raises(ValueError, match="at least 2"):
        ModelComparisonWorkflow(simulator=[make_simulator([prior, likelihood])])


def test_default_adapter_structure():
    """default_adapter() produces an Adapter with correct transform chains."""
    from bayesflow.adapters import Adapter

    # Minimal adapter (model_indices → inference_variables only)
    adapter = ModelComparisonWorkflow.default_adapter(
        inference_conditions=None,
        summary_variables=None,
    )
    assert isinstance(adapter, Adapter)

    # With summary variables
    adapter_sv = ModelComparisonWorkflow.default_adapter(
        inference_conditions=None,
        summary_variables=["x"],
    )
    assert isinstance(adapter_sv, Adapter)

    # With both inference_conditions and summary_variables
    adapter_full = ModelComparisonWorkflow.default_adapter(
        inference_conditions=["n"],
        summary_variables=["x"],
    )
    assert isinstance(adapter_full, Adapter)


def test_disabled_methods_raise(mc_simulators):
    """sample(), log_prob(), and ancestral_sample() are not supported."""
    workflow = ModelComparisonWorkflow(simulator=mc_simulators)

    with pytest.raises(NotImplementedError):
        workflow.sample()

    with pytest.raises(NotImplementedError):
        workflow.log_prob()

    with pytest.raises(NotImplementedError):
        workflow.ancestral_sample()


def test_plot_diagnostics_with_presimulated_data(mc_simulators):
    """plot_default_diagnostics accepts a pre-simulated dict instead of an int."""
    workflow = ModelComparisonWorkflow(
        simulator=mc_simulators,
        inference_conditions=["x"],
    )
    workflow.fit_online(epochs=1, batch_size=2, num_batches_per_epoch=2, verbose=0)

    test_data = workflow.simulate(15)
    plots = workflow.plot_default_diagnostics(test_data=test_data)

    assert "confusion_and_bayes_factors" in plots
    # loss plot is only added when history is available (which it is here)
    assert "loss" in plots


def test_plot_diagnostics_without_simulator_raises():
    """plot_default_diagnostics(int) raises when no simulator is attached."""
    from bayesflow.approximators import ModelComparisonApproximator
    from bayesflow.networks import MLP

    from bayesflow.networks import ScoringRuleNetwork
    from bayesflow.scoring_rules import CrossEntropyScore

    approximator = ModelComparisonApproximator(
        inference_network=ScoringRuleNetwork(
            scoring_rules={"scoring_rule": CrossEntropyScore()}, subnet=MLP(widths=(8,))
        ),
    )
    workflow = ModelComparisonWorkflow.__new__(ModelComparisonWorkflow)
    workflow.approximator = approximator
    workflow.simulator = None
    workflow.model_names = None
    workflow.history = None

    with pytest.raises(ValueError, match="No simulator"):
        workflow.plot_default_diagnostics(test_data=10)


def test_compute_diagnostics_without_simulator_raises():
    """compute_default_diagnostics(int) raises when no simulator is attached."""
    from bayesflow.approximators import ModelComparisonApproximator
    from bayesflow.networks import MLP, ScoringRuleNetwork
    from bayesflow.scoring_rules import CrossEntropyScore

    approximator = ModelComparisonApproximator(
        inference_network=ScoringRuleNetwork(
            scoring_rules={"scoring_rule": CrossEntropyScore()}, subnet=MLP(widths=(8,))
        ),
    )
    workflow = ModelComparisonWorkflow.__new__(ModelComparisonWorkflow)
    workflow.approximator = approximator
    workflow.simulator = None
    workflow.model_names = None
    workflow.history = None

    with pytest.raises(ValueError, match="No simulator"):
        workflow.compute_default_diagnostics(test_data=10)


@pytest.mark.parametrize("rule_names", [["cross_entropy", "brier"], ["exponential", "logistic"]])
def test_plot_diagnostics_multi_rule(mc_simulators, rule_names):
    """plot_default_diagnostics produces the combined figure + calibration for any multi-rule config.

    Without a ``true_log_bfs_fn`` there is no recovery plot.
    """
    workflow = ModelComparisonWorkflow(
        simulator=mc_simulators,
        scoring_rules=rule_names,
        inference_conditions=["x"],
    )
    workflow.fit_online(epochs=2, batch_size=4, num_batches_per_epoch=2, verbose=0)

    plots = workflow.plot_default_diagnostics(test_data=10)
    assert set(plots.keys()) == {"loss", "confusion_and_bayes_factors", "calibration"}


def test_compute_diagnostics_multi_rule(mc_simulators):
    """compute_default_diagnostics works for multiple rules via merged model_probs."""
    from bayesflow.scoring_rules import CrossEntropyScore, BrierScore

    workflow = ModelComparisonWorkflow(
        simulator=mc_simulators,
        scoring_rules={"ce": CrossEntropyScore(), "brier": BrierScore()},
        inference_conditions=["x"],
    )
    workflow.fit_online(epochs=2, batch_size=4, num_batches_per_epoch=2, verbose=0)

    metrics = workflow.compute_default_diagnostics(test_data=10)
    assert isinstance(metrics, pd.DataFrame)
    assert {"Accuracy", "Expected Calibration Error"} <= set(metrics.index)
    assert ((metrics.loc["Accuracy"] >= 0.0) & (metrics.loc["Accuracy"] <= 1.0)).all()


def test_plot_diagnostics_with_inference_variables_key(mc_simulators):
    """plot_default_diagnostics accepts test_data with 'inference_variables' instead of 'model_indices'."""
    workflow = ModelComparisonWorkflow(
        simulator=mc_simulators,
        inference_conditions=["x"],
    )
    workflow.fit_online(epochs=2, batch_size=4, num_batches_per_epoch=2, verbose=0)

    test_data = workflow.simulate(10)
    # Simulate pre-adapted data: rename model_indices → inference_variables
    test_data["inference_variables"] = test_data.pop("model_indices")

    plots = workflow.plot_default_diagnostics(test_data=test_data)
    assert "confusion_and_bayes_factors" in plots


def test_compute_diagnostics_with_inference_variables_key(mc_simulators):
    """compute_default_diagnostics accepts test_data with 'inference_variables' instead of 'model_indices'."""
    workflow = ModelComparisonWorkflow(
        simulator=mc_simulators,
        inference_conditions=["x"],
    )
    workflow.fit_online(epochs=1, batch_size=4, num_batches_per_epoch=2, verbose=0)

    test_data = workflow.simulate(10)
    test_data["inference_variables"] = test_data.pop("model_indices")

    metrics = workflow.compute_default_diagnostics(test_data=test_data)
    assert isinstance(metrics, pd.DataFrame)
    assert "Accuracy" in metrics.index


def test_plot_diagnostics_raises_without_model_key(mc_simulators):
    """plot_default_diagnostics raises KeyError when test_data has neither model key."""
    workflow = ModelComparisonWorkflow(
        simulator=mc_simulators,
        inference_conditions=["x"],
    )
    workflow.fit_online(epochs=1, batch_size=2, num_batches_per_epoch=2, verbose=0)

    test_data = workflow.simulate(10)
    del test_data["model_indices"]

    with pytest.raises(KeyError):
        workflow.plot_default_diagnostics(test_data=test_data)


def test_compute_diagnostics_raises_without_model_key(mc_simulators):
    """compute_default_diagnostics raises KeyError when test_data has neither model key."""
    workflow = ModelComparisonWorkflow(
        simulator=mc_simulators,
        inference_conditions=["x"],
    )
    workflow.fit_online(epochs=1, batch_size=2, num_batches_per_epoch=2, verbose=0)

    test_data = workflow.simulate(10)
    del test_data["model_indices"]

    with pytest.raises(KeyError):
        workflow.compute_default_diagnostics(test_data=test_data)
