import pytest
import keras

from bayesflow.networks import TimeSeriesTransformer

from .conftest import (
    BATCH,
    SUMMARY_DIM,
    make_3d_input,
    check_output_shape,
    check_serialize_deserialize,
    check_save_and_load,
    check_variable_set_size,
)


def _make(return_sequences=False, **kwargs):
    return TimeSeriesTransformer(
        summary_dim=SUMMARY_DIM,
        embed_dims=(8, 8),
        num_heads=(2, 2),
        dropout=0.0,
        return_sequences=return_sequences,
        **kwargs,
    )


@pytest.fixture
def net():
    return _make()


@pytest.fixture
def x():
    return make_3d_input(set_size=12)


def test_output_shape(net, x):
    check_output_shape(net, x)


def test_serialize_deserialize(net, x):
    check_serialize_deserialize(net, x)


def test_save_and_load(net, x, tmp_path):
    check_save_and_load(net, x, tmp_path)


def test_variable_sequence_length(net, x):
    check_variable_set_size(net, x)


def test_save_and_load_gru_embedding(tmp_path):
    net = _make(time_embedding="gru", time_embed_dim=4)
    check_save_and_load(net, make_3d_input(set_size=12), tmp_path)


def test_save_and_load_no_time_embedding(tmp_path):
    net = _make(time_embedding=None)
    check_save_and_load(net, make_3d_input(set_size=12), tmp_path)


def test_save_and_load_explicit_time_axis(tmp_path):
    net = _make(time_axis=2, time_embed_dim=4)
    check_save_and_load(net, make_3d_input(set_size=12, features=3), tmp_path)


@pytest.mark.parametrize("time_embedding", ["time2vec", "gru"])
def test_time_embedding_types(time_embedding):
    net = _make(time_embedding=time_embedding, time_embed_dim=4)
    x = make_3d_input(set_size=12)
    y = net(x, training=False)
    assert keras.ops.shape(y) == (BATCH, SUMMARY_DIM)


def test_no_time_embedding():
    net = _make(time_embedding=None)
    x = make_3d_input(set_size=12)
    y = net(x, training=False)
    assert keras.ops.shape(y) == (BATCH, SUMMARY_DIM)


def test_explicit_time_axis():
    net = _make(time_axis=2, time_embed_dim=4)
    x = make_3d_input(set_size=12, features=3)
    y = net(x, training=False)
    assert keras.ops.shape(y) == (BATCH, SUMMARY_DIM)


def test_return_sequences_false():
    net = _make(return_sequences=False)
    x = make_3d_input(set_size=12)
    y = net(x, training=False)
    assert len(keras.ops.shape(y)) == 2
    assert keras.ops.shape(y)[-1] == SUMMARY_DIM


@pytest.mark.parametrize("glu_variant", ["swiglu", "geglu"])
def test_glu_variants(glu_variant):
    net = _make(glu_variant=glu_variant)
    x = make_3d_input(set_size=12)
    y = net(x, training=False)
    assert keras.ops.shape(y) == (BATCH, SUMMARY_DIM)


def test_return_sequences_true():
    net = _make(return_sequences=True)
    x = make_3d_input(set_size=12)
    y = net(x, training=False)
    # many-to-many: (batch, seq_len, summary_dim)
    assert keras.ops.shape(y) == (BATCH, 12, SUMMARY_DIM)


def test_save_and_load_return_sequences_true(tmp_path):
    net = _make(return_sequences=True)
    check_save_and_load(net, make_3d_input(set_size=12), tmp_path)


def test_no_layer_norm():
    net = _make(layer_norm=False)
    y = net(make_3d_input(set_size=12), training=False)
    assert keras.ops.shape(y) == (BATCH, SUMMARY_DIM)


def test_lstm_embedding():
    net = _make(time_embedding="lstm", time_embed_dim=4)
    y = net(make_3d_input(set_size=12), training=False)
    assert keras.ops.shape(y) == (BATCH, SUMMARY_DIM)


def test_use_bias():
    net = _make(use_bias=True)
    y = net(make_3d_input(set_size=12), training=False)
    assert keras.ops.shape(y) == (BATCH, SUMMARY_DIM)
