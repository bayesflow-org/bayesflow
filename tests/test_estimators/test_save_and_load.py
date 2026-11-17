import keras
from tests.utils import assert_models_equal


def test_save_and_load(tmp_path, estimator, train_dataset):
    from bayesflow import EnsembleEstimator, EnsembleDataset

    if isinstance(estimator, EnsembleEstimator):
        train_dataset = EnsembleDataset(train_dataset, member_names=list(estimator.estimators.keys()))

    # to save, the model must be built
    data = keras.tree.map_structure(keras.ops.convert_to_tensor, train_dataset[0])
    data_shapes = keras.tree.map_structure(keras.ops.shape, data)
    estimator.build(data_shapes)
    estimator.compute_metrics(**data)

    keras.saving.save_model(estimator, tmp_path / "model.keras")
    loaded = keras.saving.load_model(tmp_path / "model.keras")

    assert_models_equal(estimator, loaded)
