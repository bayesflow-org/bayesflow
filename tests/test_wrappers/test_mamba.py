import pytest

import bayesflow as bf


def should_skip():
    import keras

    if keras.backend.backend() != "torch":
        return True, "Mamba tests can only be run on PyTorch."

    import torch

    if not torch.cuda.is_available():
        return True, "Mamba tests can only be run on GPU."

    try:
        import mamba_ssm  # noqa: F401
    except ImportError:
        return True, "Could not import mamba."

    return False, None


skip, reason = should_skip()


@pytest.mark.skipif(skip, reason=reason)
def test_mamba_summary(random_time_series, mamba_summary_network):
    out = mamba_summary_network(random_time_series)
    # Batch size 2, summary dim 4
    assert out.shape == (2, 4)


@pytest.mark.skipif(skip, reason=reason)
def test_mamba_trains(random_time_series, inference_network, mamba_summary_network):
    workflow = bf.BasicWorkflow(
        inference_network=inference_network,
        summary_network=mamba_summary_network,
        inference_variables=["parameters"],
        summary_variables=["observables"],
        simulator=bf.simulators.SIR(),
    )

    history = workflow.fit_online(epochs=2, batch_size=8, num_batches_per_epoch=2)
    assert "loss" in list(history.history.keys())
    assert len(history.history["loss"]) == 2
