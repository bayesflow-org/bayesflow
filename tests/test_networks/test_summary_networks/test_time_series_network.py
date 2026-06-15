import pytest
import keras

from bayesflow.networks import TimeSeriesNetwork

from .conftest import (
    BATCH,
    SUMMARY_DIM,
    make_3d_input,
    check_output_shape,
    check_serialize_deserialize,
    check_save_and_load,
    check_variable_set_size,
)


def _make(**kwargs):
    return TimeSeriesNetwork(
        summary_dim=SUMMARY_DIM,
        filters=(8,),
        kernel_sizes=(3,),
        strides=(1,),
        recurrent_dim=8,
        dropout=0.0,
        **kwargs,
    )


@pytest.fixture
def net():
    return _make()


@pytest.fixture
def x():
    return make_3d_input(set_size=8)


def test_output_shape(net, x):
    check_output_shape(net, x)


def test_serialize_deserialize(net, x):
    check_serialize_deserialize(net, x)


def test_save_and_load(net, x, tmp_path):
    check_save_and_load(net, x, tmp_path)


def test_variable_set_size(net, x):
    check_variable_set_size(net, x)


def test_save_and_load_unidirectional(tmp_path):
    net = _make(bidirectional=False)
    check_save_and_load(net, make_3d_input(set_size=8), tmp_path)


def test_save_and_load_lstm(tmp_path):
    net = _make(recurrent_type="lstm")
    check_save_and_load(net, make_3d_input(set_size=8), tmp_path)


def test_save_and_load_multi_scale(tmp_path):
    net = TimeSeriesNetwork(
        summary_dim=SUMMARY_DIM,
        filters=(8, 16),
        kernel_sizes=(3, 5),
        strides=(1, 1),
        recurrent_dim=8,
        dropout=0.0,
    )
    check_save_and_load(net, make_3d_input(set_size=8), tmp_path)


@pytest.mark.parametrize("recurrent_type", ["gru", "lstm"])
def test_recurrent_types(recurrent_type):
    net = _make(recurrent_type=recurrent_type)
    y = net(make_3d_input(set_size=8), training=False)
    assert keras.ops.shape(y) == (BATCH, SUMMARY_DIM)


@pytest.mark.parametrize("bidirectional", [True, False])
def test_bidirectional(bidirectional):
    net = _make(bidirectional=bidirectional)
    y = net(make_3d_input(set_size=8), training=False)
    assert keras.ops.shape(y) == (BATCH, SUMMARY_DIM)


def test_multi_scale_convolution():
    net = TimeSeriesNetwork(
        summary_dim=SUMMARY_DIM,
        filters=(8, 16),
        kernel_sizes=(3, 5),
        strides=(1, 1),
        recurrent_dim=8,
        dropout=0.0,
    )
    y = net(make_3d_input(set_size=8), training=False)
    assert keras.ops.shape(y) == (BATCH, SUMMARY_DIM)


def test_strided_convolution():
    net = TimeSeriesNetwork(
        summary_dim=SUMMARY_DIM,
        filters=(8,),
        kernel_sizes=(3,),
        strides=(2,),
        recurrent_dim=8,
        dropout=0.0,
    )
    y = net(make_3d_input(set_size=8), training=False)
    assert keras.ops.shape(y) == (BATCH, SUMMARY_DIM)


def test_scalar_broadcast():
    """Scalar filters/kernel_sizes/strides should broadcast into single-stage lists."""
    net = TimeSeriesNetwork(
        summary_dim=SUMMARY_DIM,
        filters=8,
        kernel_sizes=3,
        strides=1,
        recurrent_dim=8,
        dropout=0.0,
    )
    y = net(make_3d_input(set_size=8), training=False)
    assert keras.ops.shape(y) == (BATCH, SUMMARY_DIM)


def test_skip_steps():
    net = _make(skip_steps=2)
    y = net(make_3d_input(set_size=8), training=False)
    assert keras.ops.shape(y) == (BATCH, SUMMARY_DIM)


def test_groups_norm():
    net = _make(groups=2)
    y = net(make_3d_input(set_size=8), training=False)
    assert keras.ops.shape(y) == (BATCH, SUMMARY_DIM)


def test_custom_activation():
    net = _make(activation="relu")
    y = net(make_3d_input(set_size=8), training=False)
    assert keras.ops.shape(y) == (BATCH, SUMMARY_DIM)
