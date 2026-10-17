import keras

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
