import keras

from bayesflow.networks import ConvolutionalNetwork
from bayesflow.utils.serialization import serialize, deserialize
from tests.utils import assert_layers_equal

BATCH = 2
H, W, C = 8, 12, 3
SUMMARY_DIM = 5


def _make_network():
    return ConvolutionalNetwork(
        summary_dim=SUMMARY_DIM,
        widths=(8, 16),
        blocks_per_stage=1,
        downsample_stage=(True, False),
    )


def _input():
    return keras.random.normal((BATCH, H, W, C))


def test_output_shape():
    net = _make_network()
    x = _input()
    y = net(x, training=False)
    assert keras.ops.shape(y) == (BATCH, SUMMARY_DIM)


def test_serialize_deserialize():
    net = _make_network()
    net(_input())  # build

    serialized = serialize(net)
    deserialized = deserialize(serialized)
    reserialized = serialize(deserialized)

    assert keras.tree.lists_to_tuples(serialized) == keras.tree.lists_to_tuples(reserialized)


def test_save_and_load(tmp_path):
    net = _make_network()
    net(_input())  # build

    keras.saving.save_model(net, tmp_path / "model.keras")
    loaded = keras.saving.load_model(tmp_path / "model.keras")

    assert_layers_equal(net, loaded)
