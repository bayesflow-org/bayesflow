import os

import keras
import numpy as np
import pytest

from bayesflow.workflows import ModelComparisonWorkflow
from tests.utils import assert_models_equal


# ── PMP scoring rules ─────────────────────────────────────────────────────────


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

    # PMP diagnostics: confusion_matrix + calibration (+ loss curve)
    assert set(plots.keys()) == {"loss", "confusion_matrix", "calibration"}

    # PMP metrics: accuracy (float in [0, 1]) + per-model ECE dict
    assert "accuracy" in metrics
    assert "ece" in metrics
    assert 0.0 <= metrics["accuracy"] <= 1.0

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

    assert "confusion_matrix" in plots
    assert "calibration" in plots


# ── Bayes factor scoring rules ────────────────────────────────────────────────


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

    # BF diagnostics: calibration + pairwise_bayes_factors (+ loss); no PMP-only plots
    assert "loss" in plots
    assert "calibration" in plots
    assert "pairwise_bayes_factors" in plots
    assert "confusion_matrix" not in plots
    assert "blind_coverage" not in plots
    # no true_log_bfs_fn supplied → no recovery plot
    assert "bayes_factor_recovery" not in plots

    # BF metrics: accuracy only (no ECE for BF rules)
    assert "accuracy" in metrics
    assert "ece" not in metrics
    assert 0.0 <= metrics["accuracy"] <= 1.0

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
    assert "pairwise_bayes_factors" in plots


def test_bf_workflow_with_summary_network(mc_simulators, mc_summary_network):
    """BF workflow with a summary network."""
    from bayesflow.links import Leaky
    from bayesflow.scoring_rules import ExponentialScore

    workflow = ModelComparisonWorkflow(
        simulator=mc_simulators,
        scoring_rules=ExponentialScore(links={"log_bayes_factors": Leaky(power=2.0)}),
        summary_network=mc_summary_network,
        summary_variables=["x"],
    )
    workflow.fit_online(epochs=2, batch_size=4, num_batches_per_epoch=2, verbose=0)
    plots = workflow.plot_default_diagnostics(test_data=20)

    assert "calibration" in plots
    assert "pairwise_bayes_factors" in plots


# ── Shared simulator ──────────────────────────────────────────────────────────


def test_mc_workflow_with_shared_simulator(mc_summary_network):
    """Workflow with shared_simulator broadcasts context variables to the batch."""
    from bayesflow import make_simulator

    def shared(batch_size):
        return dict(n=np.int32(8))

    def prior_null():
        return dict(mu=0.0)

    def prior_alt():
        return dict(mu=np.random.normal(0, 1))

    def likelihood(mu, n):
        return dict(x=np.random.normal(mu, 1, n))

    simulators = [
        make_simulator([prior_null, likelihood]),
        make_simulator([prior_alt, likelihood]),
    ]

    workflow = ModelComparisonWorkflow(
        simulator=simulators,
        shared_simulator=shared,
        summary_network=mc_summary_network,
        summary_variables=["x"],
        inference_conditions=["n"],
    )
    workflow.fit_online(epochs=2, batch_size=4, num_batches_per_epoch=2, verbose=0)
    plots = workflow.plot_default_diagnostics(test_data=20)

    assert "confusion_matrix" in plots
    assert "calibration" in plots


# ── Output shapes ─────────────────────────────────────────────────────────────


def test_estimate_shapes_pmp(mc_simulators):
    """PMP mode: estimate returns model_probs (N, M) summing to 1 and logits (N, M)."""
    from bayesflow.scoring_rules import CrossEntropyScore

    num_models = len(mc_simulators)
    n_test = 10

    workflow = ModelComparisonWorkflow(
        simulator=mc_simulators,
        scoring_rules=CrossEntropyScore(),
        inference_conditions=["x"],
    )
    workflow.fit_online(epochs=2, batch_size=4, num_batches_per_epoch=2, verbose=0)
    test_data = workflow.simulate(n_test)

    estimates = workflow.estimate(conditions=test_data)

    assert isinstance(estimates, dict)
    assert estimates["model_probs"].shape == (n_test, num_models)
    assert estimates["logits"].shape == (n_test, num_models)
    assert np.allclose(estimates["model_probs"].sum(axis=-1), 1.0, atol=1e-5)


def test_estimate_shapes_bf(mc_simulators):
    """BF mode: estimate returns model_probs (N, M) summing to 1 and log_bayes_factors (N, M-1)."""
    from bayesflow.scoring_rules import ExponentialScore

    num_models = len(mc_simulators)
    n_test = 10

    workflow = ModelComparisonWorkflow(
        simulator=mc_simulators,
        scoring_rules=ExponentialScore(),
        inference_conditions=["x"],
    )
    workflow.fit_online(epochs=2, batch_size=4, num_batches_per_epoch=2, verbose=0)
    test_data = workflow.simulate(n_test)

    estimates = workflow.estimate(conditions=test_data)

    assert isinstance(estimates, dict)
    assert estimates["model_probs"].shape == (n_test, num_models)
    assert estimates["log_bayes_factors"].shape == (n_test, num_models - 1)
    assert np.allclose(estimates["model_probs"].sum(axis=-1), 1.0, atol=1e-5)


# ── Constructor validation ────────────────────────────────────────────────────


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
        broadcast_conditions_to="x",
    )
    assert isinstance(adapter_full, Adapter)


# ── Disabled BasicWorkflow methods ────────────────────────────────────────────


def test_disabled_methods_raise(mc_simulators):
    """sample(), log_prob(), and ancestral_sample() are not supported."""
    workflow = ModelComparisonWorkflow(simulator=mc_simulators)

    with pytest.raises(NotImplementedError):
        workflow.sample()

    with pytest.raises(NotImplementedError):
        workflow.log_prob()

    with pytest.raises(NotImplementedError):
        workflow.ancestral_sample()


# ── plot_default_diagnostics with pre-simulated data ─────────────────────────


def test_plot_diagnostics_with_presimulated_data(mc_simulators):
    """plot_default_diagnostics accepts a pre-simulated dict instead of an int."""
    workflow = ModelComparisonWorkflow(
        simulator=mc_simulators,
        inference_conditions=["x"],
    )
    workflow.fit_online(epochs=2, batch_size=4, num_batches_per_epoch=2, verbose=0)

    test_data = workflow.simulate(15)
    plots = workflow.plot_default_diagnostics(test_data=test_data)

    assert "confusion_matrix" in plots
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


# ── classifier_network property ───────────────────────────────────────────────


def test_plot_diagnostics_multi_rule_raises(mc_simulators):
    """plot_default_diagnostics raises NotImplementedError when multiple scoring rules are used."""
    from bayesflow.scoring_rules import CrossEntropyScore, BrierScore

    workflow = ModelComparisonWorkflow(
        simulator=mc_simulators,
        scoring_rules={"ce": CrossEntropyScore(), "brier": BrierScore()},
        inference_conditions=["x"],
    )
    workflow.fit_online(epochs=2, batch_size=4, num_batches_per_epoch=2, verbose=0)

    with pytest.raises(NotImplementedError, match="multiple scoring rules"):
        workflow.plot_default_diagnostics(test_data=10)


def test_compute_diagnostics_multi_rule_raises(mc_simulators):
    """compute_default_diagnostics raises NotImplementedError when multiple scoring rules are used."""
    from bayesflow.scoring_rules import CrossEntropyScore, BrierScore

    workflow = ModelComparisonWorkflow(
        simulator=mc_simulators,
        scoring_rules={"ce": CrossEntropyScore(), "brier": BrierScore()},
        inference_conditions=["x"],
    )
    workflow.fit_online(epochs=2, batch_size=4, num_batches_per_epoch=2, verbose=0)

    with pytest.raises(NotImplementedError, match="multiple scoring rules"):
        workflow.compute_default_diagnostics(test_data=10)


def test_classifier_network_property(mc_simulators):
    """classifier_network returns the subnet of the internal ScoringRuleNetwork."""
    workflow = ModelComparisonWorkflow(simulator=mc_simulators)
    assert workflow.classifier_network is not None
    assert workflow.classifier_network is workflow.approximator.inference_network.subnet


# ── inference_variables key branch ───────────────────────────────────────────


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
    assert "confusion_matrix" in plots


def test_compute_diagnostics_with_inference_variables_key(mc_simulators):
    """compute_default_diagnostics accepts test_data with 'inference_variables' instead of 'model_indices'."""
    workflow = ModelComparisonWorkflow(
        simulator=mc_simulators,
        inference_conditions=["x"],
    )
    workflow.fit_online(epochs=2, batch_size=4, num_batches_per_epoch=2, verbose=0)

    test_data = workflow.simulate(10)
    test_data["inference_variables"] = test_data.pop("model_indices")

    metrics = workflow.compute_default_diagnostics(test_data=test_data)
    assert "accuracy" in metrics


def test_plot_diagnostics_raises_without_model_key(mc_simulators):
    """plot_default_diagnostics raises KeyError when test_data has neither model key."""
    workflow = ModelComparisonWorkflow(
        simulator=mc_simulators,
        inference_conditions=["x"],
    )
    workflow.fit_online(epochs=2, batch_size=4, num_batches_per_epoch=2, verbose=0)

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
    workflow.fit_online(epochs=2, batch_size=4, num_batches_per_epoch=2, verbose=0)

    test_data = workflow.simulate(10)
    del test_data["model_indices"]

    with pytest.raises(KeyError):
        workflow.compute_default_diagnostics(test_data=test_data)
