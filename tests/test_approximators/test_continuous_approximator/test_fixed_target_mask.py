import keras
import numpy as np
import pytest

import bayesflow as bf
from tests.utils import assert_allclose

TOL = 1e-5
SHAPES = {"inference_variables": (8, 4), "inference_conditions": (8, 3)}


@pytest.fixture()
def data():
    theta = keras.random.normal(SHAPES["inference_variables"], seed=0)
    x = keras.random.normal(SHAPES["inference_conditions"], seed=1)
    # the last two parameters are fixed (e.g. padded) in the first half of the batch
    mask = np.ones(SHAPES["inference_variables"], dtype="float32")
    mask[:4, 2:] = 0
    return theta, x, mask


def test_loss_uses_mask(inference_network, data):
    theta, x, mask = data
    approximator = bf.ContinuousApproximator(inference_network=inference_network, standardize=None)
    approximator.build(SHAPES)

    metrics = approximator.compute_metrics(
        inference_variables=theta, inference_conditions=x, inference_fixed_target_mask=mask, stage="validation"
    )
    expected_loss = -keras.ops.mean(inference_network.log_prob(theta, conditions=x, fixed_target_mask=mask))

    assert_allclose(metrics["loss"], expected_loss, rtol=TOL, atol=TOL)


def test_log_prob_uses_mask_from_adapter(inference_network, data):
    theta, x, mask = data
    adapter = bf.ContinuousApproximator.build_adapter(
        inference_variables="theta", inference_conditions="x", inference_fixed_target_mask="theta_mask"
    )
    approximator = bf.ContinuousApproximator(inference_network=inference_network, adapter=adapter, standardize=None)
    approximator.build(SHAPES)

    inputs = {"theta": keras.ops.convert_to_numpy(theta), "x": keras.ops.convert_to_numpy(x), "theta_mask": mask}
    log_prob = approximator.log_prob(inputs)
    expected_log_prob = inference_network.log_prob(theta, conditions=x, fixed_target_mask=mask)

    assert_allclose(log_prob, expected_log_prob, rtol=TOL, atol=TOL)
