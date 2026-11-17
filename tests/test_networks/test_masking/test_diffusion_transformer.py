import inspect

import keras
import numpy as np

from bayesflow.networks import DiffusionTransformer
from bayesflow.networks.subnets.transformer.diffusion_transformer import DiffusionTransformerBlock


def _tensor(values):
    return keras.ops.convert_to_tensor(np.asarray(values, dtype=np.float32))


def _numpy(value):
    return keras.ops.convert_to_numpy(value)


class RecordingBlock(keras.Layer):
    def __init__(self):
        super().__init__()
        self.attention_mask = None
        self.update_mask = None

    def call(
        self,
        inputs,
        conditions=None,
        attention_mask=None,
        update_mask=None,
        training=None,
    ):
        self.attention_mask = attention_mask
        self.update_mask = update_mask
        return inputs[0]


def _recording_transformer():
    transformer = DiffusionTransformer(widths=(8,), num_heads=2, dropout=0.0)
    block = RecordingBlock()
    transformer.blocks = [block]
    return transformer, block


def test_attention_mask_is_an_explicit_subnet_argument():
    parameters = inspect.signature(DiffusionTransformer.call).parameters
    assert "attention_mask" in parameters


def test_explicit_and_semantic_attention_masks_are_intersected():
    transformer, block = _recording_transformer()
    x = _tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]])
    time = _tensor([[0.25], [0.75]])
    conditions = _tensor([[1.0, -1.0], [0.5, 2.0]])
    fixed = _tensor([[1.0, 0.0, 1.0], [1.0, 0.0, 1.0]])
    inferred = keras.ops.ones_like(x)
    observed = _tensor([[1.0, 0.0], [1.0, 0.0]])
    explicit = np.ones((2, 3, 5), dtype=np.bool_)
    explicit[:, 0, 0] = False
    explicit = keras.ops.convert_to_tensor(explicit)

    transformer(
        (x, time, conditions),
        attention_mask=explicit,
        fixed_target_mask=fixed,
        infer_target_mask=inferred,
        observed_condition_mask=observed,
    )

    semantic = transformer.conditioning_attention_mask(
        fixed_target_mask=fixed,
        observed_condition_mask=observed,
        infer_target_mask=inferred,
        num_targets=3,
        num_conditions=2,
        batch_size=2,
    )
    expected = keras.ops.logical_and(explicit, semantic)
    np.testing.assert_array_equal(_numpy(block.attention_mask), _numpy(expected))
    assert tuple(block.update_mask.shape) == (2, 3, 1)


def test_explicit_attention_mask_is_used_unchanged_without_semantic_masks():
    transformer, block = _recording_transformer()
    x = _tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]])
    time = _tensor([[0.25], [0.75]])
    conditions = _tensor([[1.0, -1.0], [0.5, 2.0]])
    explicit = keras.ops.convert_to_tensor(np.eye(3, 5, dtype=np.bool_)[None])

    transformer((x, time, conditions), attention_mask=explicit)

    np.testing.assert_array_equal(_numpy(block.attention_mask), _numpy(explicit))
    assert block.update_mask is None


def test_conditioning_attention_mask_three_state_dependency_truth_table():
    fixed = _tensor([[1.0, 0.0, 0.0]])
    inferred = _tensor([[1.0, 1.0, 0.0]])
    observed_conditions = _tensor([[1.0, 0.0]])

    actual = DiffusionTransformer.conditioning_attention_mask(
        fixed_target_mask=fixed,
        observed_condition_mask=observed_conditions,
        infer_target_mask=inferred,
        num_targets=3,
        num_conditions=2,
        batch_size=1,
    )

    # Rows are latent, observed, and absent target queries. Columns are the
    # three target keys followed by an observed and an absent condition key.
    expected = np.asarray(
        [
            [True, True, False, True, False],
            [False, True, False, True, False],
            [False, False, True, False, False],
        ],
        dtype=np.bool_,
    )[None]
    np.testing.assert_array_equal(_numpy(actual), expected)


def test_missing_semantic_inputs_do_not_leak_placeholder_values():
    transformer = DiffusionTransformer(widths=(8, 8), num_heads=2, dropout=0.0)
    time = _tensor([[0.5]])
    fixed = _tensor([[0.0, 1.0, 1.0]])
    inferred = _tensor([[0.0, 1.0, 1.0]])
    observed = _tensor([[0.0, 1.0]])

    baseline = transformer(
        (_tensor([[0.0, 1.0, 2.0]]), time, _tensor([[0.0, 4.0]])),
        fixed_target_mask=fixed,
        infer_target_mask=inferred,
        observed_condition_mask=observed,
    )
    hidden = transformer(
        (_tensor([[np.nan, 1.0, 2.0]]), time, _tensor([[np.nan, 4.0]])),
        fixed_target_mask=fixed,
        infer_target_mask=inferred,
        observed_condition_mask=observed,
    )

    assert np.all(np.isfinite(_numpy(hidden)))
    np.testing.assert_allclose(_numpy(hidden), _numpy(baseline), rtol=0.0, atol=1e-6)


def test_zero_update_mask_keeps_transformer_tokens():
    block = DiffusionTransformerBlock(width=8, num_heads=2)
    seed = keras.random.SeedGenerator(42)
    x = keras.random.normal((2, 3, 8), dtype="float32", seed=seed)
    base_mod = keras.random.normal((2, 6 * 8), dtype="float32", seed=seed)
    update_mask = keras.ops.zeros((2, 3, 1), dtype="float32")

    out = block((x, base_mod), update_mask=update_mask)

    np.testing.assert_allclose(_numpy(out), _numpy(x), rtol=1e-5, atol=1e-6)


def test_identity_attention_mask_makes_target_tokens_independent():
    num_features = 5
    transformer = DiffusionTransformer(widths=(16, 16), num_heads=2, dropout=0.0)
    transformer.build(((1, num_features), (1, 1), None))
    rng = np.random.default_rng(5)

    for block in transformer.blocks:
        values = np.float32(0.1) * rng.standard_normal(block.ada_ln_table.shape).astype(np.float32)
        block.ada_ln_table.assign(keras.ops.convert_to_tensor(values))

    x = rng.standard_normal((1, num_features)).astype(np.float32)
    time = keras.ops.convert_to_tensor(rng.random((1, 1)).astype(np.float32))
    identity = keras.ops.convert_to_tensor(np.eye(num_features, dtype=np.float32)[None])

    def run(values):
        result = transformer((keras.ops.convert_to_tensor(values), time, None), attention_mask=identity)
        return _numpy(result)

    baseline = run(x)
    for index in range(num_features):
        perturbed = x.copy()
        perturbed[:, index] += np.float32(5.0)
        difference = np.abs(run(perturbed) - baseline)[0]
        assert np.delete(difference, index).max() < np.float32(1e-5)
