import keras
import numpy as np

from bayesflow.networks.subnets.transformer.diffusion_transformer import DiffusionTransformer, DiffusionTransformerBlock
from bayesflow.utils.serialization import deserialize, serialize

from ...utils import assert_layers_equal


def test_diffusion_transformer_serialize_deserialize(diffusion_transformer, build_shapes_time):
    diffusion_transformer.build(**build_shapes_time)

    serialized = serialize(diffusion_transformer)
    deserialized = deserialize(serialized)
    reserialized = serialize(deserialized)

    assert reserialized == serialized


def test_save_and_load_diffusion_transformer(tmp_path, diffusion_transformer, build_shapes_time):
    diffusion_transformer.build(**build_shapes_time)

    keras.saving.save_model(diffusion_transformer, tmp_path / "model.keras")
    loaded = keras.saving.load_model(tmp_path / "model.keras")

    assert_layers_equal(diffusion_transformer, loaded)


def test_diffusion_transformer_output_shape(diffusion_transformer):
    x = keras.ops.ones((4, 3))
    t = keras.ops.ones((4, 1))
    conditions = keras.ops.ones((4, 5))

    out = diffusion_transformer((x, t, conditions), target_inference_mask=keras.ops.array([[1, 0, 1]] * 4))

    assert tuple(out.shape) == (4, 3)


def test_diffusion_transformer_block_zero_update_mask_keeps_tokens():
    block = DiffusionTransformerBlock(width=8, num_heads=2)
    x = keras.random.normal((2, 3, 8))
    base_mod = keras.random.normal((2, 6 * 8))
    update_mask = keras.ops.zeros((2, 3, 1))

    out = block((x, base_mod), update_mask=update_mask)

    assert keras.ops.all(keras.ops.isclose(out, x))


def test_diffusion_transformer_identity_mask_processes_tokens_independently(num_features=5):
    """An identity dependency mask must make each output token a function of only
    its own input token (i.e. the model estimates one-dimensional marginals)."""

    tt = DiffusionTransformer(widths=(16, 16, 16), num_heads=2, dropout=0.0)
    cond_shape = None
    tt.build(((1, num_features), (1, 1), cond_shape))
    rng = np.random.default_rng(0)

    # Make the adaLN modulation time-dependent so the masking behaviour is exercised
    # beyond the small constant residual gate used at initialization.
    for block in tt.blocks:
        block.ada_ln_table.assign(
            keras.ops.convert_to_tensor(0.1 * rng.standard_normal(block.ada_ln_table.shape).astype("float32"))
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
