import keras
from tests.utils import assert_models_equal


def test_save_and_load(tmp_path, point_approximator, train_dataset, validation_dataset):
    # to save, the model must be built
    data_shapes = keras.tree.map_structure(keras.ops.shape, train_dataset[0])
    point_approximator.build(data_shapes)
    point_approximator.compute_metrics(**train_dataset[0])

    keras.saving.save_model(point_approximator, tmp_path / "model.keras")
    loaded = keras.saving.load_model(tmp_path / "model.keras")

    assert_models_equal(point_approximator, loaded)
