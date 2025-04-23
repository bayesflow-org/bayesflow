import bayesflow as bf


def test_basic_workflow(inference_network, summary_network):
    workflow = bf.BasicWorkflow(
        inference_network=inference_network,
        summary_network=summary_network,
        inference_variables=["parameters"],
        summary_variables=["observables"],
        simulator=bf.simulators.SIR(),
    )

    history = workflow.fit_online(epochs=2, batch_size=32, num_batches_per_epoch=2)
    plots = workflow.plot_default_diagnostics(test_data=50, num_samples=50)
    metrics = workflow.compute_default_diagnostics(test_data=50, num_samples=50, variable_names=["p1", "p2"])

    assert "loss" in list(history.history.keys())
    assert len(history.history["loss"]) == 2
    assert list(plots.keys()) == ["losses", "recovery", "calibration_ecdf", "z_score_contraction"]
    assert list(metrics.columns) == ["p1", "p2"]
    assert metrics.values.shape == (3, 2)
