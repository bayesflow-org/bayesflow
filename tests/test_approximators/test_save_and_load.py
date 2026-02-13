import keras
from tests.utils import assert_models_equal


def test_save_and_load(tmp_path, approximator, train_dataset):
    from bayesflow import EnsembleApproximator, EnsembleDataset

    if isinstance(approximator, EnsembleApproximator):
        train_dataset = EnsembleDataset(train_dataset, member_names=list(approximator.approximators.keys()))

    # to save, the model must be built
    data = train_dataset[0]
    data_shapes = keras.tree.map_structure(keras.ops.shape, data)
    approximator.build(data_shapes)
    approximator.compute_metrics(**data)

    keras.saving.save_model(approximator, tmp_path / "model.keras")
    loaded = keras.saving.load_model(tmp_path / "model.keras")

    assert_models_equal(approximator, loaded)
