import pytest
import os

import keras

import bayesflow as bf
from tests.utils import assert_models_equal


def test_ensemble_workflow_raises_for_invalid_ensemble_size():
    with pytest.raises(ValueError, match="ensemble_size"):
        bf.EnsembleWorkflow(
            inference_networks="coupling_flow",
            ensemble_size=1,  # invalid
        )


def test_ensemble_workflow_raises_when_no_dict_and_no_ensemble_size():
    with pytest.raises(ValueError, match="Either `inference_networks` is a dictionary"):
        bf.EnsembleWorkflow(
            inference_networks="coupling_flow",
            ensemble_size=None,
        )


def test_dict_inference_networks_ignores_ensemble_args(caplog):
    bf.EnsembleWorkflow(
        inference_networks={"a": "coupling_flow"},
        ensemble_size=5,
        share_inference_network=True,
    )

    msgs = " ".join(r.message for r in caplog.records)
    assert "Ignoring argument ensemble_size" in msgs
    assert "Ignoring argument share_inference_network" in msgs


def test_ensemble_workflow_raises_for_summary_key_without_inference_key():
    with pytest.raises(ValueError, match="summary network was specified"):
        bf.EnsembleWorkflow(
            inference_networks={"a": "coupling_flow"},
            summary_networks={"b": None},  # mismatched key
        )


def test_share_inference_network_true_reuses_same_model():
    workflow = bf.EnsembleWorkflow(
        inference_networks="coupling_flow",
        summary_networks=None,
        ensemble_size=3,
        share_inference_network=True,
    )

    members = workflow.approximator.approximators
    nets = [members[k].inference_network for k in sorted(members.keys())]

    assert nets[0] is nets[1] is nets[2]


def test_share_inference_network_false_clones_models():
    workflow = bf.EnsembleWorkflow(
        inference_networks="coupling_flow",
        summary_networks=None,
        ensemble_size=3,
        share_inference_network=False,
    )

    members = workflow.approximator.approximators
    nets = [members[k].inference_network for k in sorted(members.keys())]

    assert nets[0] is not nets[1]
    assert nets[1] is not nets[2]


def test_single_summary_network_is_broadcast_to_all_members():
    workflow = bf.EnsembleWorkflow(
        inference_networks="coupling_flow",
        summary_networks="time_series_network",
        ensemble_size=2,
        share_inference_network=False,
    )

    members = workflow.approximator.approximators
    s0 = members["0"].summary_network
    s1 = members["1"].summary_network

    # same object is assigned to all members in current implementation
    assert s0 is s1


def test_ensemble_workflow(tmp_path):
    workflow = bf.EnsembleWorkflow(
        inference_networks=dict(
            nf="coupling_flow",
            parametric=bf.networks.ScoringRuleNetwork(scoring_rules=dict(mvn=bf.scoring_rules.MvNormalScore())),
            point=bf.networks.ScoringRuleNetwork(scoring_rules=dict(mean=bf.scoring_rules.MeanScore())),
        ),
        summary_networks=dict(
            nf=bf.networks.TimeSeriesNetwork(),
            parametric=bf.networks.TimeSeriesNetwork(),
            point=bf.networks.TimeSeriesNetwork(),
        ),
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
