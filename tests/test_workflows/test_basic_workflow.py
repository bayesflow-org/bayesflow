import os

import keras
import numpy as np

import bayesflow as bf
from tests.utils import assert_models_equal


def test_basic_workflow(tmp_path, inference_network, summary_network):
    workflow = bf.BasicWorkflow(
        inference_network=inference_network,
        summary_network=summary_network,
        inference_variables=["parameters"],
        summary_variables=["observables"],
        simulator=bf.simulators.SIR(subsample=None),
        checkpoint_filepath=str(tmp_path),
    )

    # Ensure metrics work fine batched
    history = workflow.fit_online(epochs=2, batch_size=3, num_batches_per_epoch=2, verbose=0)
    plots = workflow.plot_default_diagnostics(test_data=25, num_samples=10)
    metrics = workflow.compute_default_diagnostics(test_data=10, num_samples=20, variable_names=["p1", "p2"])

    assert "loss" in list(history.history.keys())
    assert len(history.history["loss"]) == 2
    assert list(plots.keys()) == ["losses", "recovery", "calibration_ecdf", "coverage", "z_score_contraction"]
    assert list(metrics.columns) == ["p1", "p2"]
    assert metrics.values.shape == (4, 2)

    # Ensure saving and loading from workflow works fine
    loaded_approximator = keras.saving.load_model(os.path.join(str(tmp_path), "model.keras"))
    assert_models_equal(workflow.approximator, loaded_approximator)

    # Get samples (non-batched and batched)
    test_conditions = workflow.simulate(5)
    samples = loaded_approximator.sample(conditions=test_conditions, num_samples=3)
    batched_samples = loaded_approximator.sample(conditions=test_conditions, num_samples=3, batch_size=2)
    assert samples["parameters"].shape == (5, 3, 2)
    assert batched_samples["parameters"].shape == samples["parameters"].shape


def test_basic_workflow_fusion(
    tmp_path, fusion_inference_network, fusion_summary_network, fusion_simulator, fusion_adapter
):
    workflow = bf.BasicWorkflow(
        adapter=fusion_adapter,
        inference_network=fusion_inference_network,
        summary_network=fusion_summary_network,
        simulator=fusion_simulator,
        checkpoint_filepath=str(tmp_path),
    )

    # Ensure metrics work fine
    history = workflow.fit_online(epochs=2, batch_size=4, num_batches_per_epoch=2, verbose=0)
    plots = workflow.plot_default_diagnostics(test_data=50, num_samples=25)
    metrics = workflow.compute_default_diagnostics(test_data=50, num_samples=25, variable_names=["p1", "p2"])

    assert "loss" in list(history.history.keys())
    assert len(history.history["loss"]) == 2
    assert list(plots.keys()) == ["losses", "recovery", "calibration_ecdf", "coverage", "z_score_contraction"]
    assert list(metrics.columns) == ["p1", "p2"]
    assert metrics.values.shape == (4, 2)

    # Ensure saving and loading from workflow works fine
    loaded_approximator = keras.saving.load_model(os.path.join(str(tmp_path), "model.keras"))
    assert_models_equal(workflow.approximator, loaded_approximator)

    # Get samples (non-batched and batched)
    test_conditions = workflow.simulate(3)
    samples = loaded_approximator.sample(conditions=test_conditions, num_samples=4)
    batched_samples = loaded_approximator.sample(conditions=test_conditions, num_samples=4, batch_size=2)
    assert samples["mean"].shape == (3, 4, 2)
    assert batched_samples["mean"].shape == samples["mean"].shape


def test_unconditional_sampling(inference_network):
    workflow = bf.BasicWorkflow(
        inference_network=inference_network,
        inference_variables=["parameters"],
        simulator=bf.simulators.TwoMoons(),
    )
    workflow.fit_online(epochs=2, batch_size=3, num_batches_per_epoch=2, verbose=0)

    # Get samples
    samples = workflow.sample(conditions=None, num_samples=3)
    assert samples["parameters"].shape == (3, 2)


def test_ancestral_sampling():
    """
    Hierarchical scenario:
      global:  mu ~ N(0, 1)                      (shared across subjects)
      local:   beta_i ~ N(mu, 0.1),  x_i = beta_i + noise

    The local-level workflow approximates p(beta | mu, x).
    Global posterior samples are passed as ancestral_conditions; per-subject
    observations as conditions. The function should expand both and return
    samples of shape (n_test, n_subjects, num_samples, 1).
    """
    from bayesflow.networks import ConsistencyModel
    from bayesflow import BasicWorkflow
    from bayesflow.simulators import Simulator

    class LocalSimulator(Simulator):
        def sample(self, n, local_n=1, **kwargs):
            mu = np.random.randn(n, 1)
            beta = mu + 0.1 * np.random.randn(n, local_n)
            x = beta + 0.1 * np.random.randn(n, local_n)
            return {"mu": mu, "beta": beta, "x": x}

    workflow = BasicWorkflow(
        inference_network=ConsistencyModel(subnet_kwargs=dict(widths=(8, 8)), total_steps=5 * 2),
        inference_variables=["beta"],
        inference_conditions=["mu", "x"],
        simulator=LocalSimulator(),
    )
    workflow.fit_online(epochs=5, batch_size=4, num_batches_per_epoch=2, verbose=0)

    n_test = 3
    local_n = 4
    num_samples = 2

    # Global posterior samples from an upper-level approximator (pre-computed)
    ancestral_conditions = {"mu": np.random.randn(n_test, num_samples, 1)}
    # Per-subject local observations
    conditions = {"x": np.random.randn(n_test, local_n, 1)}

    samples = workflow.ancestral_sample(
        conditions=conditions,
        ancestral_conditions=ancestral_conditions,
    )

    assert "beta" in samples
    assert samples["beta"].shape == (n_test, local_n, num_samples, 1)
