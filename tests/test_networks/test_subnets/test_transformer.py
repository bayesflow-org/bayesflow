import keras
import numpy as np
import pytest
import tensorflow as tf

from bayesflow.networks.subnets.time_transformer import TimeTransformer, TimeTransformerBlock
from bayesflow.utils.serialization import deserialize, serialize

from ...utils import assert_layers_equal


def test_time_transformer_serialize_deserialize(time_transformer, build_shapes_time):
    time_transformer.build(**build_shapes_time)

    serialized = serialize(time_transformer)
    deserialized = deserialize(serialized)
    reserialized = serialize(deserialized)

    assert reserialized == serialized


def test_save_and_load_time_transformer(tmp_path, time_transformer, build_shapes_time):
    time_transformer.build(**build_shapes_time)

    keras.saving.save_model(time_transformer, tmp_path / "model.keras")
    loaded = keras.saving.load_model(tmp_path / "model.keras")

    assert_layers_equal(time_transformer, loaded)


def test_time_transformer_output_shape(time_transformer):
    x = keras.ops.ones((4, 3))
    t = keras.ops.ones((4, 1))
    conditions = keras.ops.ones((4, 5))

    out = time_transformer((x, t, conditions), target_inference_mask=keras.ops.array([[1, 0, 1]] * 4))

    assert tuple(out.shape) == (4, 3)


def test_time_transformer_block_zero_update_mask_keeps_tokens():
    block = TimeTransformerBlock(width=8, num_heads=2)
    x = keras.random.normal((2, 3, 8))
    t = keras.random.normal((2, 4))
    update_mask = keras.ops.zeros((2, 3, 1))

    out = block((x, t), update_mask=update_mask)

    assert keras.ops.all(keras.ops.isclose(out, x))


def test_time_transformer_rejects_non_divisible_heads():
    with pytest.raises(ValueError, match="divisible"):
        TimeTransformer(widths=(10, 10), num_heads=4)


def test_time_transformer_uses_lean_ffn_width_by_default():
    tt = TimeTransformer(widths=(16,), num_heads=2, expansion_factor=4.0)

    assert tt.blocks[0].ffn.intermediate_dim == 42


def test_time_transformer_block_residual_branches_get_initial_gradients():
    block = TimeTransformerBlock(width=8, num_heads=2, dropout=0.0, residual_gate_init=1e-2)
    x = keras.random.normal((2, 3, 8))
    t = keras.random.normal((2, 4))

    block.build((x.shape, t.shape))
    bias = keras.ops.convert_to_numpy(block.ada_ln.bias)

    assert np.allclose(bias[2 * block.width : 3 * block.width], 1e-2)
    assert np.allclose(bias[5 * block.width : 6 * block.width], 1e-2)

    with tf.GradientTape() as tape:
        out = block((x, t), training=True)
        loss = keras.ops.mean(keras.ops.square(out))

    variables = block.attn.trainable_variables + block.ffn.trainable_variables
    grads = tape.gradient(loss, variables)
    grad_norm = sum(float(keras.ops.convert_to_numpy(keras.ops.sum(keras.ops.abs(g)))) for g in grads if g is not None)

    assert grad_norm > 0.0


def test_time_transformer_identity_mask_processes_tokens_independently(num_features=5):
    """An identity dependency mask must make each output token a function of only
    its own input token (i.e. the model estimates one-dimensional marginals)."""

    tt = TimeTransformer(widths=(16, 16, 16), num_heads=2, dropout=0.0)
    cond_shape = None
    tt.build(((1, num_features), (1, 1), cond_shape))
    rng = np.random.default_rng(0)

    # Make the adaLN modulation time-dependent so the masking behaviour is exercised
    # beyond the small constant residual gate used at initialization.
    for block in tt.blocks:
        block.ada_ln.kernel.assign(
            keras.ops.convert_to_tensor(0.1 * rng.standard_normal(block.ada_ln.kernel.shape).astype("float32"))
        )

    rng = np.random.default_rng(1)
    x = rng.standard_normal((1, 5)).astype("float32")
    t = keras.ops.convert_to_tensor(rng.random((1, 1)).astype("float32"))
    identity = keras.ops.convert_to_tensor(np.eye(5, dtype="float32")[None])

    def run(arr):
        out = tt((keras.ops.convert_to_tensor(arr), t, None), attention_mask=identity)
        return keras.ops.convert_to_numpy(out)

    base = run(x)
    for j in range(5):
        perturbed = x.copy()
        perturbed[:, j] += 5.0
        diff = np.abs(run(perturbed) - base)[0]
        off_token = np.delete(diff, j)
        assert off_token.max() < 1e-5, f"identity mask leaked from token {j} to others"
