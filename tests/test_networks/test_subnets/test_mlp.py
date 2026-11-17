import keras
import pytest

from bayesflow.networks import TimeMLP
from bayesflow.utils.serialization import deserialize, serialize

from ...utils import assert_allclose, assert_layers_equal


def test_serialize_deserialize(mlp, build_shapes):
    mlp.build(**build_shapes)

    serialized = serialize(mlp)
    deserialized = deserialize(serialized)
    reserialized = serialize(deserialized)

    assert reserialized == serialized


def test_save_and_load(tmp_path, mlp, build_shapes):
    mlp.build(**build_shapes)

    keras.saving.save_model(mlp, tmp_path / "model.keras")
    loaded = keras.saving.load_model(tmp_path / "model.keras")

    assert_layers_equal(mlp, loaded)


def test_save_and_load_time_mlp(tmp_path, time_mlp, build_shapes_time):
    time_mlp.build(**build_shapes_time)

    keras.saving.save_model(time_mlp, tmp_path / "model.keras")
    loaded = keras.saving.load_model(tmp_path / "model.keras")

    assert_layers_equal(time_mlp, loaded)


@pytest.mark.parametrize("persistent_conditioning", [False, True])
@pytest.mark.parametrize("use_conditions", [False, True])
def test_time_mlp_embed_and_blocks_reproduce_call(build_shapes_time, persistent_conditioning, use_conditions):
    x_shape, t_shape, conditions_shape = build_shapes_time["input_shape"]
    input_shape = (x_shape, t_shape, conditions_shape if use_conditions else None)
    time_mlp = TimeMLP(widths=[16, 16], persistent_conditioning=persistent_conditioning)
    time_mlp.build(input_shape)

    conditions = keras.random.normal(conditions_shape) if use_conditions else None
    inputs = (keras.random.normal(x_shape), keras.random.uniform(t_shape), conditions)
    h, block_condition = time_mlp.embed(inputs, training=False)

    assert (tuple(h.shape), tuple(block_condition.shape)) == time_mlp.compute_embed_output_shape(input_shape)

    for block in time_mlp.blocks:
        h = block((h, block_condition), training=False)

    assert_allclose(h, time_mlp(inputs, training=False))
