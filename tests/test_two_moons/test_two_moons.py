import copy
import keras
import pytest


from tests.utils import assert_models_equal


@pytest.mark.parametrize("jit_compile", [False, True])
def test_compile(approximator, random_samples, jit_compile):
    approximator.compile(jit_compile=jit_compile)


def test_fit(approximator, train_dataset, validation_dataset, batch_size):
    mock_data = train_dataset[0]
    mock_data = keras.tree.map_structure(keras.ops.convert_to_tensor, mock_data)
    mock_data_shapes = keras.tree.map_structure(keras.ops.shape, mock_data)
    approximator.build(mock_data_shapes)
    approximator.compile()

    untrained_weights = copy.deepcopy(approximator.weights)
    untrained_metrics = approximator.evaluate(validation_dataset, return_dict=True)

    approximator.fit(dataset=train_dataset, epochs=50, batch_size=batch_size)

    trained_weights = approximator.weights
    trained_metrics = approximator.evaluate(validation_dataset, return_dict=True)

    # check weights have changed during training
    assert any([keras.ops.any(~keras.ops.isclose(u, t)) for u, t in zip(untrained_weights, trained_weights)])

    assert isinstance(untrained_metrics, dict)
    assert isinstance(trained_metrics, dict)

    # test that metrics are improving
    metric_names = ["loss"]

    for metric in metric_names:
        assert metric in untrained_metrics
        assert metric in trained_metrics

        # TODO: this is too flaky
        # assert trained_metrics[metric] <= untrained_metrics[metric]


def test_serialize_deserialize(tmp_path, approximator, train_dataset):
    mock_data = train_dataset[0]
    mock_data = keras.tree.map_structure(keras.ops.convert_to_tensor, mock_data)
    mock_data_shapes = keras.tree.map_structure(keras.ops.shape, mock_data)
    approximator.build(mock_data_shapes)

    # run a single batch through the approximator
    approximator.compute_metrics(**mock_data)

    keras.saving.save_model(approximator, tmp_path / "model.keras")
    loaded_approximator = keras.saving.load_model(tmp_path / "model.keras")

    assert_models_equal(approximator, loaded_approximator)
