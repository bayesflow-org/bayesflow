import keras

from bayesflow.utils.serialization import deserialize, serialize

from ...utils import assert_layers_equal


def test_serialize_deserialize(residual, build_shapes):
    residual.build(**build_shapes)

    serialized = serialize(residual)
    deserialized = deserialize(serialized)
    reserialized = serialize(deserialized)

    assert reserialized == serialized


def test_save_and_load(tmp_path, residual, build_shapes):
    residual.build(**build_shapes)

    keras.saving.save_model(residual, tmp_path / "model.keras")
    loaded = keras.saving.load_model(tmp_path / "model.keras")

    assert_layers_equal(residual, loaded)


def test_save_and_load_cond(tmp_path, cond_residual, build_shapes):
    cond_residual.build(**build_shapes)

    keras.saving.save_model(cond_residual, tmp_path / "model.keras")
    loaded = keras.saving.load_model(tmp_path / "model.keras")

    assert_layers_equal(cond_residual, loaded)
