from bayesflow.utils.serialization import deserialize, serialize
import pytest
import keras

from tests.utils import assert_layers_equal, allclose


@pytest.mark.parametrize("automatic", [True, False])
def test_build(automatic, multimodal_summary_network, multimodal_data):
    if multimodal_summary_network is None:
        pytest.skip(reason="Nothing to do, because there is no summary network.")

    assert multimodal_summary_network.built is False

    if automatic:
        multimodal_summary_network(multimodal_data)
    else:
        multimodal_summary_network.build(keras.tree.map_structure(keras.ops.shape, multimodal_data))

    assert multimodal_summary_network.built is True

    # check the model has variables
    assert multimodal_summary_network.variables, "Model has no variables."


@pytest.mark.parametrize("automatic", [True, False])
def test_build_functional_api(automatic, multimodal_summary_network, multimodal_data):
    if multimodal_summary_network is None:
        pytest.skip(reason="Nothing to do, because there is no summary network.")

    assert multimodal_summary_network.built is False

    inputs = {}
    for k, v in multimodal_data.items():
        inputs[k] = keras.layers.Input(shape=keras.ops.shape(v)[1:], name=k)
    outputs = multimodal_summary_network(inputs)
    model = keras.Model(inputs=inputs, outputs=outputs)

    if automatic:
        model(multimodal_data)
    else:
        model.build(keras.tree.map_structure(keras.ops.shape, multimodal_data))

    assert model.built is True

    # check the model has variables
    assert multimodal_summary_network.variables, "Model has no variables."


def test_serialize_deserialize(multimodal_summary_network, multimodal_data):
    if multimodal_summary_network is None:
        pytest.skip(reason="Nothing to do, because there is no summary network.")

    multimodal_summary_network.build(keras.tree.map_structure(keras.ops.shape, multimodal_data))

    serialized = serialize(multimodal_summary_network)
    deserialized = deserialize(serialized)
    reserialized = serialize(deserialized)

    assert keras.tree.lists_to_tuples(serialized) == keras.tree.lists_to_tuples(reserialized)


def test_save_and_load(tmp_path, multimodal_summary_network, multimodal_data):
    if multimodal_summary_network is None:
        pytest.skip(reason="Nothing to do, because there is no summary network.")

    multimodal_summary_network.build(keras.tree.map_structure(keras.ops.shape, multimodal_data))

    keras.saving.save_model(multimodal_summary_network, tmp_path / "model.keras")
    loaded = keras.saving.load_model(tmp_path / "model.keras")

    assert_layers_equal(multimodal_summary_network, loaded)
    assert allclose(multimodal_summary_network(multimodal_data), loaded(multimodal_data))


@pytest.mark.parametrize("stage", ["training", "validation"])
def test_compute_metrics(stage, multimodal_summary_network, multimodal_data):
    if multimodal_summary_network is None:
        pytest.skip("Nothing to do, because there is no summary network.")

    multimodal_summary_network.build(keras.tree.map_structure(keras.ops.shape, multimodal_data))

    metrics = multimodal_summary_network.compute_metrics(multimodal_data, stage=stage)
    outputs_via_call = multimodal_summary_network(multimodal_data, training=stage == "training")

    assert "outputs" in metrics

    # check that call and compute_metrics give equal outputs
    if stage != "training":
        assert allclose(metrics["outputs"], outputs_via_call)

    # check that the batch dimension is preserved
    assert (
        keras.ops.shape(metrics["outputs"])[0]
        == keras.ops.shape(multimodal_data[next(iter(multimodal_data.keys()))])[0]
    )

    assert "loss" in metrics
    assert keras.ops.shape(metrics["loss"]) == ()

    if stage != "training":
        for metric in multimodal_summary_network.metrics:
            assert metric.name in metrics
            assert keras.ops.shape(metrics[metric.name]) == ()
