import keras
from tests.utils import assert_models_equal


def test_save_and_load(tmp_path, estimator, train_dataset, validation_dataset):
    # to save, the model must be built
    data_shapes = keras.tree.map_structure(keras.ops.shape, train_dataset[0])
    estimator.build(data_shapes)
    for layer in estimator.standardizer.standardize_layers.values():
        assert layer.built
        for count in layer.count:
            assert count == 0.0
    estimator.compute_metrics(**train_dataset[0])

    keras.saving.save_model(estimator, tmp_path / "model.keras")
    loaded = keras.saving.load_model(tmp_path / "model.keras")

    assert_models_equal(estimator, loaded)
