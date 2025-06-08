import keras
import numpy as np
import io
from contextlib import redirect_stdout

from tests.utils import assert_models_equal


def test_build(approximator, train_dataset):
    assert approximator.built is False

    data_shapes = keras.tree.map_structure(keras.ops.shape, train_dataset[0])
    approximator.build(data_shapes)

    assert approximator.built is True
    assert approximator.classifier_network.built is True
    if approximator.summary_network is not None:
        assert approximator.summary_network.built is True


def test_build_adapter():
    from bayesflow.approximators import ModelComparisonApproximator

    _ = ModelComparisonApproximator.build_adapter(
        num_models=2,
        classifier_conditions=["foo", "bar"],
        summary_variables=["observables"],
        model_index_name=["indices"],
    )


def test_build_dataset(simulator, adapter):
    from bayesflow.approximators import ModelComparisonApproximator
    from bayesflow.datasets import OnlineDataset

    dataset = ModelComparisonApproximator.build_dataset(
        simulator=simulator,
        memory_budget="20 KiB",
        num_batches=2,
        num_models=2,
        classifier_conditions="foo",
        summary_variables=["x1", "x2"],
    )
    assert isinstance(dataset, OnlineDataset)


def test_fit(approximator, train_dataset, validation_dataset):
    approximator.compile(optimizer="AdamW")
    num_epochs = 1

    # Capture ostream and train model
    with io.StringIO() as stream:
        with redirect_stdout(stream):
            approximator.fit(dataset=train_dataset, validation_data=validation_dataset, epochs=num_epochs)

        output = stream.getvalue()
    # check that the loss is shown
    assert "loss" in output


def test_save_and_load(tmp_path, approximator, train_dataset, validation_dataset):
    # to save, the model must be built
    data_shapes = keras.tree.map_structure(keras.ops.shape, train_dataset[0])
    approximator.build(data_shapes)
    approximator.compute_metrics(**train_dataset[0])

    keras.saving.save_model(approximator, tmp_path / "model.keras")
    loaded = keras.saving.load_model(tmp_path / "model.keras")

    assert_models_equal(approximator, loaded)


def test_predict(approximator, train_dataset, simulator):
    data_shapes = keras.tree.map_structure(keras.ops.shape, train_dataset[0])
    approximator.build(data_shapes)
    approximator.compute_metrics(**train_dataset[0])

    num_conditions = 2
    conditions = simulator.sample(num_conditions)
    output = approximator.predict(conditions=conditions)
    assert isinstance(output, np.ndarray)
    assert output.shape[0] == num_conditions
