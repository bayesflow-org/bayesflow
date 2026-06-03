"""Shared helpers for summary-network tests."""

import keras

from bayesflow.utils.serialization import deserialize, serialize
from tests.utils import assert_layers_equal

BATCH = 2
SET_SIZE = 6
FEATURE_DIM = 3
SUMMARY_DIM = 4


def make_3d_input(batch=BATCH, set_size=SET_SIZE, features=FEATURE_DIM):
    return keras.random.normal((batch, set_size, features))


def check_output_shape(net, x, expected_dim=SUMMARY_DIM):
    y = net(x, training=False)
    assert keras.ops.shape(y) == (keras.ops.shape(x)[0], expected_dim)


def check_serialize_deserialize(net, x):
    net(x)  # build
    serialized = serialize(net)
    deserialized = deserialize(serialized)
    reserialized = serialize(deserialized)
    assert keras.tree.lists_to_tuples(serialized) == keras.tree.lists_to_tuples(reserialized)


def check_save_and_load(net, x, tmp_path):
    net(x)  # build
    keras.saving.save_model(net, tmp_path / "model.keras")
    loaded = keras.saving.load_model(tmp_path / "model.keras")
    assert_layers_equal(net, loaded)


def check_variable_set_size(net, x):
    """Confirm the network generalises to unseen set/sequence sizes."""
    net(x)  # build at one size
    for new_size in [3, 5]:
        new_x = keras.random.normal((BATCH, new_size, keras.ops.shape(x)[-1]))
        net(new_x, training=False)
