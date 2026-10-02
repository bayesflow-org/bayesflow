import keras
import pytest

import bayesflow as bf


def test_inference_mask_raises_for_coupling_flow():
    """coupling flows use no masks, so `inference_mask` must raise."""
    approximator = bf.ContinuousApproximator(inference_network=bf.networks.CouplingFlow(), standardize=None)
    approximator.build({"inference_variables": (8, 4), "inference_conditions": (8, 3)})

    with pytest.raises(ValueError, match="'mask'"):
        approximator.compute_metrics(
            inference_variables=keras.random.normal((8, 4)),
            inference_conditions=keras.random.normal((8, 3)),
            inference_mask=keras.ops.ones((8, 4)),
        )
