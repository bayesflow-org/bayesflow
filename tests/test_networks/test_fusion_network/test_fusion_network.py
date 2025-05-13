from bayesflow.utils.serialization import deserialize, serialize
import pytest
import keras

from tests.utils import assert_layers_equal, allclose


@pytest.mark.parametrize("automatic", [True, False])
def test_build(automatic, fusion_network, multimodal_data):
    if fusion_network is None:
        pytest.skip(reason="Nothing to do, because there is no summary network.")

    assert fusion_network.built is False

    if automatic:
        fusion_network(multimodal_data)
    else:
        fusion_network.build(keras.tree.map_structure(keras.ops.shape, multimodal_data))

    assert fusion_network.built is True

    # check the model has variables
    assert fusion_network.variables, "Model has no variables."


@pytest.mark.parametrize("automatic", [True, False])
def test_build_functional_api(automatic, fusion_network, multimodal_data):
    if fusion_network is None:
        pytest.skip(reason="Nothing to do, because there is no summary network.")

    assert fusion_network.built is False

    inputs = {}
    for k, v in multimodal_data.items():
        inputs[k] = keras.layers.Input(shape=keras.ops.shape(v)[1:], name=k)
    outputs = fusion_network(inputs)
    model = keras.Model(inputs=inputs, outputs=outputs)

    if automatic:
        model(multimodal_data)
    else:
        model.build(keras.tree.map_structure(keras.ops.shape, multimodal_data))

    assert model.built is True

    # check the model has variables
    assert fusion_network.variables, "Model has no variables."


def test_serialize_deserialize(fusion_network, multimodal_data):
    if fusion_network is None:
        pytest.skip(reason="Nothing to do, because there is no summary network.")

    fusion_network.build(keras.tree.map_structure(keras.ops.shape, multimodal_data))

    serialized = serialize(fusion_network)
    deserialized = deserialize(serialized)
    reserialized = serialize(deserialized)

    assert keras.tree.lists_to_tuples(serialized) == keras.tree.lists_to_tuples(reserialized)


def test_save_and_load(tmp_path, fusion_network, multimodal_data):
    if fusion_network is None:
        pytest.skip(reason="Nothing to do, because there is no summary network.")

    fusion_network.build(keras.tree.map_structure(keras.ops.shape, multimodal_data))

    keras.saving.save_model(fusion_network, tmp_path / "model.keras")
    loaded = keras.saving.load_model(tmp_path / "model.keras")

    assert_layers_equal(fusion_network, loaded)
    assert allclose(fusion_network(multimodal_data), loaded(multimodal_data))


@pytest.mark.parametrize("stage", ["training", "validation"])
def test_compute_metrics(stage, fusion_network, multimodal_data):
    if fusion_network is None:
        pytest.skip("Nothing to do, because there is no summary network.")

    fusion_network.build(keras.tree.map_structure(keras.ops.shape, multimodal_data))

    metrics = fusion_network.compute_metrics(multimodal_data, stage=stage)
    outputs_via_call = fusion_network(multimodal_data, training=stage == "training")

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
        for metric in fusion_network.metrics:
            assert metric.name in metrics
            assert keras.ops.shape(metrics[metric.name]) == ()
