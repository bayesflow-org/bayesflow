import pytest
import keras

from bayesflow.networks import FusionTransformer

from .conftest import (
    BATCH,
    SUMMARY_DIM,
    make_3d_input,
    check_output_shape,
    check_serialize_deserialize,
    check_save_and_load,
    check_variable_set_size,
)


def _make(template_dim=8, **kwargs):
    return FusionTransformer(
        summary_dim=SUMMARY_DIM,
        embed_dims=(8, 8),
        num_heads=(2, 2),
        dropout=0.0,
        template_dim=template_dim,
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


def test_save_and_load_gru_template(tmp_path):
    net = _make(template_type="gru")
    check_save_and_load(net, make_3d_input(set_size=9), tmp_path)


def test_save_and_load_unidirectional(tmp_path):
    net = _make(bidirectional=False)
    check_save_and_load(net, make_3d_input(set_size=9), tmp_path)


@pytest.mark.parametrize("template_type", ["lstm", "gru"])
def test_template_types(template_type):
    net = _make(template_type=template_type)
    y = net(make_3d_input(set_size=9), training=False)
    assert keras.ops.shape(y) == (BATCH, SUMMARY_DIM)


@pytest.mark.parametrize("bidirectional", [True, False])
def test_bidirectional(bidirectional):
    net = _make(bidirectional=bidirectional)
    y = net(make_3d_input(set_size=9), training=False)
    assert keras.ops.shape(y) == (BATCH, SUMMARY_DIM)


def test_no_layer_norm():
    net = _make(layer_norm=False)
    y = net(make_3d_input(set_size=9), training=False)
    assert keras.ops.shape(y) == (BATCH, SUMMARY_DIM)


@pytest.mark.parametrize("glu_variant", ["swiglu", "liglu"])
def test_glu_variants(glu_variant):
    net = _make(glu_variant=glu_variant)
    y = net(make_3d_input(set_size=9), training=False)
    assert keras.ops.shape(y) == (BATCH, SUMMARY_DIM)


def test_use_bias():
    net = _make(use_bias=True)
    y = net(make_3d_input(set_size=9), training=False)
    assert keras.ops.shape(y) == (BATCH, SUMMARY_DIM)


def test_custom_template_dim():
    net = _make(template_dim=16)
    y = net(make_3d_input(set_size=9), training=False)
    assert keras.ops.shape(y) == (BATCH, SUMMARY_DIM)


def test_save_and_load_no_layer_norm(tmp_path):
    net = _make(layer_norm=False)
    check_save_and_load(net, make_3d_input(set_size=9), tmp_path)
