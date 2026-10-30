import keras
import numpy as np
import pytest

from bayesflow.masking import InferenceMasks
from bayesflow.networks.subnets.transformer import DiffusionTransformerReadout

from .conftest import make_transformer_mean_flow, max_abs_diff


def test_transformer_velocity_readout(random_samples, random_conditions):
    network = make_transformer_mean_flow(random_samples, random_conditions)
    assert type(network.velocity_readout) is DiffusionTransformerReadout

    readout_weights = {id(w) for w in network.velocity_readout.weights}
    assert readout_weights and readout_weights.isdisjoint(id(w) for w in network.subnet.weights)


def test_transformer_reads_target_time(random_samples, random_conditions):
    network = make_transformer_mean_flow(random_samples, random_conditions)
    u = network._average_velocity(random_samples, 0.8, 0.8, random_conditions, False, InferenceMasks())
    u_moved = network._average_velocity(random_samples, 0.8, 0.2, random_conditions, False, InferenceMasks())

    assert np.std(keras.ops.convert_to_numpy(u)) > 1e-3
    assert max_abs_diff(u, u_moved) > 1e-4


def test_transformer_compute_metrics(random_samples, random_conditions):
    if keras.backend.backend() == "torch":
        import torch

        if not torch.cuda.is_available():
            pytest.skip("PyTorch CPU attention does not support the forward-mode AD used by mean flows.")

    network = make_transformer_mean_flow(random_samples, random_conditions)
    loss = keras.ops.convert_to_numpy(network.compute_metrics(random_samples, conditions=random_conditions)["loss"])
    assert np.isfinite(loss), f"Loss is not finite: {loss}"
