import keras
from tests.utils import assert_models_equal


def test_save_and_load(tmp_path, approximator, train_dataset):
    from bayesflow import EnsembleApproximator, EnsembleDataset

    if isinstance(approximator, EnsembleApproximator):
        train_dataset = EnsembleDataset(train_dataset, num_ensemble=len(approximator.approximators))

    # to save, the model must be built
    approximator.build_from_data(train_dataset[0])
    approximator.compute_metrics(**train_dataset[0])

    keras.saving.save_model(approximator, tmp_path / "model.keras")
    loaded = keras.saving.load_model(tmp_path / "model.keras")

    assert_models_equal(approximator, loaded)
