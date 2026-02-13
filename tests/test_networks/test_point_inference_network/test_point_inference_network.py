import keras
from keras.saving import (
    deserialize_keras_object as deserialize,
    serialize_keras_object as serialize,
)
from tests.utils import assert_layers_equal
import pytest


def test_output_structure(scoring_rule_inference_network, random_samples, random_conditions):
    output = scoring_rule_inference_network(random_samples, conditions=random_conditions)

    assert isinstance(output, dict)
    for score_key, score in scoring_rule_inference_network.scores.items():
        head_shapes = score.get_head_shapes_from_target_shape(random_samples.shape)
        assert isinstance(head_shapes, dict)

        for head_key, head_shape in head_shapes.items():
            head_output = output[score_key][head_key]
            assert keras.ops.is_tensor(head_output)
            assert head_output.shape[1:] == head_shape


def test_serialize_deserialize(scoring_rule_inference_network, random_samples, random_conditions):
    # to save, the model must be built
    scoring_rule_inference_network(random_samples, conditions=random_conditions)

    serialized = serialize(scoring_rule_inference_network)
    deserialized = deserialize(serialized)
    reserialized = serialize(deserialized)

    assert serialized == reserialized


def test_save_and_load(tmp_path, scoring_rule_inference_network, random_samples, random_conditions):
    # to save, the model must be built
    out1 = scoring_rule_inference_network(random_samples, conditions=random_conditions)

    keras.saving.save_model(scoring_rule_inference_network, tmp_path / "model.keras")
    loaded = keras.saving.load_model(tmp_path / "model.keras")
    out2 = loaded(random_samples, conditions=random_conditions)

    assert_layers_equal(scoring_rule_inference_network, loaded)

    for key_outer in out1.keys():
        for key_inner in out1[key_outer].keys():
            assert keras.ops.all(keras.ops.isclose(out1[key_outer][key_inner], out2[key_outer][key_inner])), (
                "Output of original and loaded model differs significantly."
            )


def test_copy_unequal(scoring_rule_inference_network, random_samples, random_conditions):
    # to save, the model must be built
    scoring_rule_inference_network(random_samples, conditions=random_conditions)

    copied = keras.models.clone_model(scoring_rule_inference_network)

    with pytest.raises(AssertionError) as excinfo:
        assert_layers_equal(scoring_rule_inference_network, copied)

    assert "not equal" in str(excinfo)


def test_save_and_load_quantile(tmp_path, quantile_scoring_rule_inference_network, random_samples, random_conditions):
    """Test of all nested attributes for a point inference network with a quantile head"""

    # to save, the model must be built
    net = quantile_scoring_rule_inference_network
    net(random_samples, conditions=random_conditions)

    keras.saving.save_model(net, tmp_path / "model.keras")
    loaded = keras.saving.load_model(tmp_path / "model.keras")

    print(net.get_config())
    assert net.get_config() == loaded.get_config()

    assert_layers_equal(net, loaded)

    for score_key, score in net.scores.items():
        for head_key, head in net.heads[score_key].items():
            net_head = net.heads[score_key][head_key]
            loaded_head = loaded.heads[score_key][head_key]

            net_score = net.scores[score_key]
            loaded_score = loaded.scores[score_key]

            assert keras.ops.all(keras.ops.isclose(net_score._q, loaded_score._q))
            assert keras.ops.all(keras.ops.isclose(net_head.layers[-1].q, loaded_head.layers[-1].q))
            assert keras.ops.all(net_head.layers[-1].anchor_index == loaded_head.layers[-1].anchor_index)

            print(f"Asserting original and serialized and deserialized at heads[{score_key}][{head_key}] to be equal")
            assert_layers_equal(net_head, loaded_head)
