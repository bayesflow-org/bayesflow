from tests.utils import assert_models_equal
import keras
import pytest

from bayesflow.experimental.self_consistency import SelfConsistencyLoss


@pytest.mark.tensorflow
def test_metrics(dataset, sc_estimator):
    # test whether compile defines the correct metrics for correct datasets
    sc_estimator.compile(
        keras.optimizers.Adam(),
        estimator_metrics={"labeled": ["posterior"]},
        composite_metrics={"unlabeled": SelfConsistencyLoss(num_samples=16)},
    )
    history = sc_estimator.fit(dataset=dataset, epochs=1)
    assert set(history.history.keys()) == {"loss", "labeled/posterior/loss", "unlabeled/self-consistency/loss"}

    sc_estimator.compile(
        keras.optimizers.Adam(),
        composite_metrics={
            "labeled": SelfConsistencyLoss(num_samples=16),
            "unlabeled": SelfConsistencyLoss(num_samples=16),
        },
    )

    history = sc_estimator.fit(dataset=dataset, epochs=1)
    assert set(history.history.keys()) == {"loss", "labeled/self-consistency/loss", "unlabeled/self-consistency/loss"}


@pytest.mark.tensorflow
def test_save_and_load(tmp_path, dataset, sc_estimator):
    sc_estimator.compile(
        keras.optimizers.Adam(),
        estimator_metrics={"labeled": ["posterior"]},
        composite_metrics={"unlabeled": SelfConsistencyLoss(num_samples=16)},
    )
    sc_estimator.fit(dataset=dataset, epochs=1)

    keras.saving.save_model(sc_estimator, tmp_path / "model.keras")
    loaded = keras.saving.load_model(tmp_path / "model.keras")

    assert_models_equal(sc_estimator, loaded)
