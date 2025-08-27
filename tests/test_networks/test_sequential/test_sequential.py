import keras

from bayesflow.utils.serialization import deserialize, serialize

from ...utils import assert_layers_equal


def test_serialize_deserialize(sequential, build_shapes):
    sequential.build(**build_shapes)

    serialized = serialize(sequential)
    deserialized = deserialize(serialized)
    reserialized = serialize(deserialized)

    assert reserialized == serialized


def test_save_and_load(tmp_path, sequential, build_shapes):
    sequential.build(**build_shapes)

    keras.saving.save_model(sequential, tmp_path / "model.keras")
    loaded = keras.saving.load_model(tmp_path / "model.keras")

    assert_layers_equal(sequential, loaded)
