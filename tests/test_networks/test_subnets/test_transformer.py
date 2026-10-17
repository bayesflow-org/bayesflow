import keras
import numpy as np
import pytest

from bayesflow.networks import DiffusionTransformer
from bayesflow.networks.helpers import DiffusionTransformerReadout
from bayesflow.utils.serialization import deserialize, serialize

from ...utils import assert_allclose, assert_layers_equal


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


def _random_diffusion_transformer(input_shape):
    """A built transformer whose zero-initialized output layers are randomized, so its output is not zero."""
    transformer = DiffusionTransformer(widths=[16, 16], num_heads=4, dropout=0.0)
    transformer.build(input_shape)
    rng = np.random.default_rng(0)
    for weight in transformer.weights:
        values = 0.1 * rng.standard_normal(tuple(weight.shape)).astype(np.float32)
        weight.assign(keras.ops.convert_to_tensor(values))
    return transformer


def _transformer_inputs(input_shape):
    rng = np.random.default_rng(1)
    return tuple(keras.ops.convert_to_tensor(rng.random(shape).astype(np.float32)) for shape in input_shape)


@pytest.mark.parametrize("use_masks", [False, True])
def test_diffusion_transformer_embed_blocks_and_readout_reproduce_call(use_masks):
    input_shape = ((2, 3), (2, 1), (2, 2))
    transformer = _random_diffusion_transformer(input_shape)
    inputs = _transformer_inputs(input_shape)

    masks = {}
    if use_masks:
        attention_mask = np.ones((2, 3, 5), dtype=bool)
        attention_mask[:, 0, -1] = False
        masks = {
            "attention_mask": keras.ops.convert_to_tensor(attention_mask),
            "fixed_target_mask": keras.ops.convert_to_tensor(np.float32([[1, 0, 1], [1, 1, 0]])),
            "infer_target_mask": keras.ops.convert_to_tensor(np.float32([[1, 1, 1], [1, 1, 0]])),
            "observed_condition_mask": keras.ops.convert_to_tensor(np.float32([[1, 0], [1, 1]])),
        }

    h, context = transformer.embed(inputs, training=False, **masks)

    h_shape, context_shape = transformer.compute_embed_output_shape(input_shape)
    assert tuple(h.shape) == h_shape
    for tensor, shape in zip(context, context_shape):
        # the masks are None without mask inputs
        if tensor is not None:
            assert tuple(tensor.shape) == tuple(shape)

    for block in transformer.blocks:
        h = block((h, context), training=False)
    out = DiffusionTransformerReadout.final_layer(
        h, context, transformer.out_norm, transformer.final_ada_ln, transformer.token_out, training=False
    )

    assert_allclose(out, transformer(inputs, training=False, **masks))


def test_diffusion_transformer_make_readout():
    input_shape = ((2, 3), (2, 1), (2, 2))
    transformer = _random_diffusion_transformer(input_shape)
    inputs = _transformer_inputs(input_shape)

    readout = transformer.make_readout()
    assert isinstance(readout, DiffusionTransformerReadout)
    assert not readout.built
    assert readout.width == transformer.width

    h_shape, context_shape = transformer.compute_embed_output_shape(input_shape)
    readout.build((h_shape, context_shape))
    readout.final_ada_ln.set_weights(transformer.final_ada_ln.get_weights())
    readout.token_out.set_weights(transformer.token_out.get_weights())

    h, context = transformer.embed(inputs, training=False)
    for block in transformer.blocks:
        h = block((h, context), training=False)
    out = readout((h, context), training=False)

    assert tuple(out.shape) == readout.compute_output_shape((h_shape, context_shape))
    assert_allclose(out, transformer(inputs, training=False))


def test_diffusion_transformer_reads_every_time_channel():
    input_shape = ((2, 3), (2, 2), (2, 2))
    transformer = _random_diffusion_transformer(input_shape)
    x, t, conditions = _transformer_inputs(input_shape)
    t_shifted = keras.ops.concatenate([t[:, :1], t[:, 1:] + 0.5], axis=-1)

    out = transformer((x, t, conditions), training=False)
    out_shifted = transformer((x, t_shifted, conditions), training=False)

    assert not np.allclose(keras.ops.convert_to_numpy(out), keras.ops.convert_to_numpy(out_shifted))
