import pytest
import keras

from bayesflow.networks import ConvolutionalNetwork

from bayesflow.utils.serialization import deserialize, serialize
from bayesflow.distributions import DiagonalNormal
from tests.utils import assert_layers_equal

BATCH = 2
H, W, C = 8, 8, 3
SUMMARY_DIM = 4


def _make(**kwargs):
    return ConvolutionalNetwork(
        summary_dim=SUMMARY_DIM,
        widths=(4, 8),
        blocks_per_stage=1,
        downsample_stage=(True, False),
        **kwargs,
    )


def _input(h=H, w=W, c=C):
    return keras.random.normal((BATCH, h, w, c))


def test_output_shape():
    net = _make()
    y = net(_input(), training=False)
    assert keras.ops.shape(y) == (BATCH, SUMMARY_DIM)


def test_serialize_deserialize():
    net = _make()
    net(_input())

    serialized = serialize(net)
    deserialized = deserialize(serialized)
    reserialized = serialize(deserialized)

    assert keras.tree.lists_to_tuples(serialized) == keras.tree.lists_to_tuples(reserialized)


def test_save_and_load(tmp_path):
    net = _make()
    net(_input())

    keras.saving.save_model(net, tmp_path / "model.keras")
    loaded = keras.saving.load_model(tmp_path / "model.keras")
    assert_layers_equal(net, loaded)


def test_save_and_load_attention_pool(tmp_path):
    net = _make(pool_head="attention")
    net(_input())
    keras.saving.save_model(net, tmp_path / "model.keras")
    loaded = keras.saving.load_model(tmp_path / "model.keras")
    assert_layers_equal(net, loaded)


def test_save_and_load_conv_downsample(tmp_path):
    net = _make(down_mode="conv")
    net(_input())
    keras.saving.save_model(net, tmp_path / "model.keras")
    loaded = keras.saving.load_model(tmp_path / "model.keras")
    assert_layers_equal(net, loaded)


@pytest.mark.parametrize("norm", ["layer", "group", None])
def test_norm_options(norm):
    kwargs = {"norm": norm}
    if norm == "group":
        kwargs["groups"] = 2
    net = _make(**kwargs)
    y = net(_input(), training=False)
    assert keras.ops.shape(y) == (BATCH, SUMMARY_DIM)


@pytest.mark.parametrize("pool_head", ["flatten", "global_avg", "attention"])
def test_pool_head_options(pool_head):
    net = _make(pool_head=pool_head)
    y = net(_input(), training=False)
    assert keras.ops.shape(y) == (BATCH, SUMMARY_DIM)


@pytest.mark.parametrize("down_mode", ["max_pool", "conv"])
def test_downsample_modes(down_mode):
    net = _make(down_mode=down_mode)
    y = net(_input(), training=False)
    assert keras.ops.shape(y) == (BATCH, SUMMARY_DIM)


def test_no_residual():
    net = _make(residual=False)
    y = net(_input(), training=False)
    assert keras.ops.shape(y) == (BATCH, SUMMARY_DIM)


def test_non_square_input():
    net = _make()
    net(_input())  # build on square
    y = net(_input(h=6, w=10), training=False)
    assert keras.ops.shape(y) == (BATCH, SUMMARY_DIM)


def test_save_and_load_flatten(tmp_path):
    net = _make(pool_head="flatten")
    net(_input())
    keras.saving.save_model(net, tmp_path / "model.keras")
    loaded = keras.saving.load_model(tmp_path / "model.keras")
    assert_layers_equal(net, loaded)


def test_multi_block_stages():
    net = ConvolutionalNetwork(
        summary_dim=SUMMARY_DIM,
        widths=(4, 8),
        kernel_sizes=(2, 2),
        blocks_per_stage=2,
        downsample_stage=(True, False),
    )
    y = net(_input(), training=False)
    assert keras.ops.shape(y) == (BATCH, SUMMARY_DIM)


def test_grayscale_input():
    net = _make()
    y = net(_input(c=1), training=False)
    assert keras.ops.shape(y) == (BATCH, SUMMARY_DIM)


def test_dropout():
    net = _make(dropout=0.1)
    y = net(_input(), training=False)
    assert keras.ops.shape(y) == (BATCH, SUMMARY_DIM)


def test_convolutional_network_forwards_kwargs():
    """
    See: https://github.com/bayesflow-org/bayesflow/issues/699
    """
    net = ConvolutionalNetwork(base_distribution="normal")
    assert isinstance(net.base_distribution, DiagonalNormal)
