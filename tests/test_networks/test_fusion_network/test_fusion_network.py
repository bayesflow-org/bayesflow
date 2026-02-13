from bayesflow.utils.serialization import deserialize, serialize
import pytest
import keras

from tests.utils import assert_layers_equal, allclose


@pytest.mark.parametrize("automatic", [True, False])
def test_build(automatic, fusion_network, data, multimodal):
    if fusion_network is None:
        pytest.skip(reason="Nothing to do, because there is no summary network.")

    assert fusion_network.built is False

    if automatic:
        fusion_network(data)
    else:
        fusion_network.build(keras.tree.map_structure(keras.ops.shape, data))

    assert fusion_network.built is True

    # check the model has variables
    assert fusion_network.variables, "Model has no variables."


def test_build_failure(fusion_network, data, multimodal):
    if not multimodal:
        pytest.skip(reason="Nothing to do, as summary networks may consume aribrary inputs")
    with pytest.raises(ValueError):
        fusion_network.build((3, 2, 2))
    with pytest.raises(ValueError):
        data["x3"] = data.pop("x1")
        fusion_network.build(keras.tree.map_structure(keras.ops.shape, data))


@pytest.mark.parametrize("automatic", [True, False])
def test_build_functional_api(automatic, fusion_network, data, multimodal):
    if fusion_network is None:
        pytest.skip(reason="Nothing to do, because there is no summary network.")

    assert fusion_network.built is False

    if multimodal:
        inputs = {}
        for k, v in data.items():
            inputs[k] = keras.layers.Input(shape=keras.ops.shape(v)[1:], name=k)
    else:
        inputs = keras.layers.Input(shape=keras.ops.shape(data)[1:])
    outputs = fusion_network(inputs)
    model = keras.Model(inputs=inputs, outputs=outputs)

    if automatic:
        model(data)
    else:
        model.build(keras.tree.map_structure(keras.ops.shape, data))

    assert model.built is True

    # check the model has variables
    assert fusion_network.variables, "Model has no variables."


def test_serialize_deserialize(fusion_network, data, multimodal):
    if fusion_network is None:
        pytest.skip(reason="Nothing to do, because there is no summary network.")

    fusion_network.build(keras.tree.map_structure(keras.ops.shape, data))

    serialized = serialize(fusion_network)
    deserialized = deserialize(serialized)
    reserialized = serialize(deserialized)

    assert keras.tree.lists_to_tuples(serialized) == keras.tree.lists_to_tuples(reserialized)


def test_save_and_load(tmp_path, fusion_network, data, multimodal):
    if fusion_network is None:
        pytest.skip(reason="Nothing to do, because there is no summary network.")

    fusion_network.build(keras.tree.map_structure(keras.ops.shape, data))

    keras.saving.save_model(fusion_network, tmp_path / "model.keras")
    loaded = keras.saving.load_model(tmp_path / "model.keras")

    assert_layers_equal(fusion_network, loaded)
    assert allclose(fusion_network(data), loaded(data))


@pytest.mark.parametrize("stage", ["training", "validation"])
def test_compute_metrics(stage, fusion_network, data, multimodal):
    if fusion_network is None:
        pytest.skip("Nothing to do, because there is no summary network.")

    fusion_network.build(keras.tree.map_structure(keras.ops.shape, data))

    metrics = fusion_network.compute_metrics(data, stage=stage)
    outputs_via_call = fusion_network(data, training=stage == "training")

    assert "outputs" in metrics

    # check that call and compute_metrics give equal outputs
    if stage != "training":
        assert allclose(metrics["outputs"], outputs_via_call)

    # check that the batch dimension is preserved
    batch_size = keras.ops.shape(data)[0] if not multimodal else keras.ops.shape(data[next(iter(data.keys()))])[0]

    assert keras.ops.shape(metrics["outputs"])[0] == batch_size
    assert "loss" in metrics
    assert keras.ops.shape(metrics["loss"]) == ()

    if stage != "training":
        for metric in fusion_network.metrics:
            assert metric.name in metrics
            assert keras.ops.shape(metrics[metric.name]) == ()
