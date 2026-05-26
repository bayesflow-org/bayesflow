import pytest
import keras

from bayesflow.networks import DeepSet

from .conftest import (
    BATCH,
    SUMMARY_DIM,
    make_3d_input,
    check_output_shape,
    check_serialize_deserialize,
    check_save_and_load,
    check_variable_set_size,
)


def _make(layer_norm=True, num_seeds=1, **kwargs):
    return DeepSet(
        summary_dim=SUMMARY_DIM,
        embed_dim=8,
        depth=1,
        mlp_widths=(8,),
        num_heads=2,
        num_seeds=num_seeds,
        dropout=0.0,
        layer_norm=layer_norm,
        **kwargs,
    )


@pytest.fixture
def net():
    return _make()


@pytest.fixture
def x():
    return make_3d_input()


def test_output_shape(net, x):
    check_output_shape(net, x)


def test_serialize_deserialize(net, x):
    check_serialize_deserialize(net, x)


def test_save_and_load(net, x, tmp_path):
    check_save_and_load(net, x, tmp_path)


def test_variable_set_size(net, x):
    check_variable_set_size(net, x)


def test_save_and_load_no_layer_norm(tmp_path):
    net = _make(layer_norm=False)
    check_save_and_load(net, make_3d_input(), tmp_path)


def test_save_and_load_multiple_seeds(tmp_path):
    net = _make(num_seeds=3)
    check_save_and_load(net, make_3d_input(), tmp_path)


def test_no_layer_norm():
    net = _make(layer_norm=False)
    y = net(make_3d_input(), training=False)
    assert keras.ops.shape(y) == (BATCH, SUMMARY_DIM)


@pytest.mark.parametrize("glu_variant", ["swiglu", "geglu"])
def test_glu_variants(glu_variant):
    net = _make(glu_variant=glu_variant)
    y = net(make_3d_input(), training=False)
    assert keras.ops.shape(y) == (BATCH, SUMMARY_DIM)


@pytest.mark.parametrize("inner_pooling", ["mean", "max"])
def test_inner_pooling_options(inner_pooling):
    net = _make(inner_pooling=inner_pooling)
    y = net(make_3d_input(), training=False)
    assert keras.ops.shape(y) == (BATCH, SUMMARY_DIM)


def test_multiple_seeds():
    net = _make(num_seeds=3)
    y = net(make_3d_input(), training=False)
    assert keras.ops.shape(y) == (BATCH, SUMMARY_DIM)


def test_deeper_stack():
    net = DeepSet(
        summary_dim=SUMMARY_DIM,
        embed_dim=8,
        depth=3,
        mlp_widths=(8,),
        num_heads=2,
        num_seeds=1,
        dropout=0.0,
    )
    y = net(make_3d_input(), training=False)
    assert keras.ops.shape(y) == (BATCH, SUMMARY_DIM)


def test_use_bias():
    net = _make(use_bias=True)
    y = net(make_3d_input(), training=False)
    assert keras.ops.shape(y) == (BATCH, SUMMARY_DIM)


def test_custom_seed_dim():
    net = _make(seed_dim=16)
    y = net(make_3d_input(), training=False)
    assert keras.ops.shape(y) == (BATCH, SUMMARY_DIM)


def test_custom_activation():
    net = _make(activation="mish")
    y = net(make_3d_input(), training=False)
    assert keras.ops.shape(y) == (BATCH, SUMMARY_DIM)
