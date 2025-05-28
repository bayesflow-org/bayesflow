import keras

from bayesflow.utils.serialization import deserialize, serialize

from ...utils import assert_layers_equal


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
