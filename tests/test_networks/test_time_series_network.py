import keras

from bayesflow.networks.summary.recurrent.time_series_network import TimeSeriesNetwork
from bayesflow.utils.serialization import serialize, deserialize
from tests.utils import assert_layers_equal

BATCH = 2
SEQ_LEN = 16
CHANNELS = 8
SUMMARY_DIM = 5


def _make_network():
    return TimeSeriesNetwork(
        summary_dim=SUMMARY_DIM,
        filters=(8, 16),
        kernel_sizes=(3, 3),
        strides=(1, 1),
        recurrent_dim=4,
        bidirectional=True,
        dropout=0.0,
    )


def _input():
    return keras.random.normal((BATCH, SEQ_LEN, CHANNELS))


def test_output_shape():
    net = _make_network()
    x = _input()
    y = net(x, training=False)
    assert keras.ops.shape(y) == (BATCH, SUMMARY_DIM)


def test_serialize_deserialize():
    net = _make_network()
    net(_input())

    serialized = serialize(net)
    deserialized = deserialize(serialized)
    reserialized = serialize(deserialized)

    assert keras.tree.lists_to_tuples(serialized) == keras.tree.lists_to_tuples(reserialized)


def test_save_and_load(tmp_path):
    net = _make_network()
    net(_input())

    keras.saving.save_model(net, tmp_path / "model.keras")
    loaded = keras.saving.load_model(tmp_path / "model.keras")

    assert_layers_equal(net, loaded)
