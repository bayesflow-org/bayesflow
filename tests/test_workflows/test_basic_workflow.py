import os

import keras

import bayesflow as bf
from tests.utils import assert_models_equal


def test_basic_workflow(tmp_path, inference_network, summary_network):
    workflow = bf.BasicWorkflow(
        inference_network=inference_network,
        summary_network=summary_network,
        inference_variables=["parameters"],
        summary_variables=["observables"],
        simulator=bf.simulators.SIR(),
        checkpoint_filepath=str(tmp_path),
    )

    # Ensure metrics work fine
    history = workflow.fit_online(epochs=4, batch_size=8, num_batches_per_epoch=2, verbose=0)
    plots = workflow.plot_default_diagnostics(test_data=50, num_samples=25)
    metrics = workflow.compute_default_diagnostics(test_data=50, num_samples=25, variable_names=["p1", "p2"])

    assert "loss" in list(history.history.keys())
    assert len(history.history["loss"]) == 4
    assert list(plots.keys()) == ["losses", "recovery", "calibration_ecdf", "z_score_contraction"]
    assert list(metrics.columns) == ["p1", "p2"]
    assert metrics.values.shape == (3, 2)

    # Ensure saving and loading from workflow works fine
    loaded_approximator = keras.saving.load_model(os.path.join(str(tmp_path), "model.keras"))
    assert_models_equal(workflow.approximator, loaded_approximator)

    # Get samples
    samples = loaded_approximator.sample(conditions=workflow.simulate(5), num_samples=3)
    assert samples["parameters"].shape == (5, 3, 2)


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
    history = workflow.fit_online(epochs=4, batch_size=8, num_batches_per_epoch=2, verbose=0)
    plots = workflow.plot_default_diagnostics(test_data=50, num_samples=25)
    metrics = workflow.compute_default_diagnostics(test_data=50, num_samples=25, variable_names=["p1", "p2"])

    assert "loss" in list(history.history.keys())
    assert len(history.history["loss"]) == 4
    assert list(plots.keys()) == ["losses", "recovery", "calibration_ecdf", "z_score_contraction"]
    assert list(metrics.columns) == ["p1", "p2"]
    assert metrics.values.shape == (3, 2)

    # Ensure saving and loading from workflow works fine
    loaded_approximator = keras.saving.load_model(os.path.join(str(tmp_path), "model.keras"))
    assert_models_equal(workflow.approximator, loaded_approximator)

    # Get samples
    samples = loaded_approximator.sample(conditions=workflow.simulate(5), num_samples=3)
    assert samples["mean"].shape == (5, 3, 2)
