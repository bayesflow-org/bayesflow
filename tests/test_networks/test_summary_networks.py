import keras
import numpy as np
import pytest

from bayesflow.utils.serialization import deserialize, serialize

from tests.utils import assert_layers_equal


@pytest.mark.parametrize("automatic", [True, False])
def test_build(automatic, summary_network, random_set):
    if summary_network is None:
        pytest.skip(reason="Nothing to do, because there is no summary network.")

    assert summary_network.built is False

    if automatic:
        summary_network(random_set)
    else:
        summary_network.build(keras.ops.shape(random_set))

    assert summary_network.built is True

    # check the model has variables
    assert summary_network.variables, "Model has no variables."


@pytest.mark.parametrize("automatic", [True, False])
def test_build_functional_api(automatic, summary_network, random_set):
    if summary_network is None:
        pytest.skip(reason="Nothing to do, because there is no summary network.")

    assert summary_network.built is False

    inputs = keras.layers.Input(shape=keras.ops.shape(random_set)[1:])
    outputs = summary_network(inputs)
    model = keras.Model(inputs=inputs, outputs=outputs)

    if automatic:
        model(random_set)
    else:
        model.build(keras.ops.shape(random_set))

    assert model.built is True

    # check the model has variables
    assert summary_network.variables, "Model has no variables."


def test_variable_batch_size(summary_network, random_set):
    if summary_network is None:
        pytest.skip(reason="Nothing to do, because there is no summary network.")

    # build with one batch size
    summary_network.build(keras.ops.shape(random_set))

    # run with another batch size
    batch_sizes = np.random.choice(10, replace=False, size=3)
    for b in batch_sizes:
        new_input = keras.ops.zeros((b,) + keras.ops.shape(random_set)[1:])
        summary_network(new_input)


def test_variable_set_size(summary_network, random_set):
    if summary_network is None:
        pytest.skip(reason="Nothing to do, because there is no summary network.")

    # build with one set size
    summary_network.build(keras.ops.shape(random_set))

    # run with another set size
    for s in [3, 4, 5]:
        b = keras.ops.shape(random_set)[0]
        new_input = keras.ops.zeros((b, s, keras.ops.shape(random_set)[2]))
        summary_network(new_input)


def test_serialize_deserialize(summary_network, random_set):
    if summary_network is None:
        pytest.skip(reason="Nothing to do, because there is no summary network.")

    summary_network.build(keras.ops.shape(random_set))

    serialized = serialize(summary_network)
    deserialized = deserialize(serialized)
    reserialized = serialize(deserialized)

    assert keras.tree.lists_to_tuples(serialized) == keras.tree.lists_to_tuples(reserialized)


def test_save_and_load(tmp_path, summary_network, random_set):
    if summary_network is None:
        pytest.skip(reason="Nothing to do, because there is no summary network.")

    summary_network.build(keras.ops.shape(random_set))

    keras.saving.save_model(summary_network, tmp_path / "model.keras")
    loaded = keras.saving.load_model(tmp_path / "model.keras")

    assert_layers_equal(summary_network, loaded)


@pytest.mark.parametrize("stage", ["training", "validation"])
def test_compute_metrics(stage, summary_network, random_set):
    if summary_network is None:
        pytest.skip("Nothing to do, because there is no summary network.")

    summary_network.build(keras.ops.shape(random_set))

    metrics = summary_network.compute_metrics(random_set, stage=stage)

    assert "outputs" in metrics

    # check that the batch dimension is preserved
    assert keras.ops.shape(metrics["outputs"])[0] == keras.ops.shape(random_set)[0]

    # check summary dimension
    summary_dim = summary_network.summary_dim
    assert keras.ops.shape(metrics["outputs"])[-1] == summary_dim

    if summary_network.base_distribution is not None:
        assert "loss" in metrics
        assert keras.ops.shape(metrics["loss"]) == ()

        if stage != "training":
            for metric in summary_network.metrics:
                assert metric.name in metrics
                assert keras.ops.shape(metrics[metric.name]) == ()
