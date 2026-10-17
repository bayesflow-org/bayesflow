import keras
import numpy as np

from bayesflow.adapters import Adapter
from bayesflow.approximators import CompositionalApproximator
from bayesflow.networks import DiffusionModel
from bayesflow.masking import MaskName


def _numpy(value):
    return keras.ops.convert_to_numpy(value)


def test_discovered_compositional_masks_keep_their_semantic_shapes():
    approximator = CompositionalApproximator(
        inference_network=DiffusionModel(
            subnet_kwargs={"widths": (8, 8)},
            integrate_kwargs={"method": "euler", "steps": 2},
        ),
        adapter=Adapter(),
        standardize=None,
    )

    inference_conditions = np.arange(12, dtype=np.float32).reshape(2, 3, 2)
    observed_condition_mask = np.asarray(
        [
            [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
            [[0.0, 1.0], [1.0, 0.0], [1.0, 1.0]],
        ],
        dtype=np.float32,
    )
    fixed_target_mask = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    infer_target_mask = np.ones((2, 2), dtype=np.float32)
    fixed_target_value = np.asarray([[2.0, 3.0], [4.0, 5.0]], dtype=np.float32)

    resolved_conditions, adapted, _ = approximator._prepare_compositional_conditions(
        conditions={
            "inference_conditions": inference_conditions,
            MaskName.OBSERVED_CONDITION: observed_condition_mask,
            MaskName.FIXED_TARGET: fixed_target_mask,
            MaskName.INFER_TARGET: infer_target_mask,
            MaskName.FIXED_TARGET_VALUE: fixed_target_value,
        }
    )
    inference_kwargs = approximator._prepare_inference_kwargs({}, adapted=adapted)

    assert tuple(resolved_conditions.shape) == (2, 3, 2)
    np.testing.assert_array_equal(_numpy(resolved_conditions), inference_conditions)
    np.testing.assert_array_equal(
        _numpy(inference_kwargs[MaskName.OBSERVED_CONDITION]),
        observed_condition_mask,
    )
    np.testing.assert_array_equal(_numpy(inference_kwargs[MaskName.FIXED_TARGET]), fixed_target_mask)
    np.testing.assert_array_equal(_numpy(inference_kwargs[MaskName.INFER_TARGET]), infer_target_mask)
    np.testing.assert_array_equal(_numpy(inference_kwargs[MaskName.FIXED_TARGET_VALUE]), fixed_target_value)

    for value in inference_kwargs.values():
        if hasattr(value, "dtype"):
            assert _numpy(value).dtype == np.float32
