"""Tests for native Keras FlashAttention integration and option propagation."""

import keras
import numpy as np
import pytest

from bayesflow.networks.summary.transformers.attention import MultiHeadAttention
from bayesflow.utils.serialization import deserialize, serialize


@pytest.fixture()
def x():
    return keras.random.normal((2, 6, 8), seed=0)


def test_uses_native_attention_with_external_dropout():
    block = MultiHeadAttention(embed_dim=8, num_heads=2, dropout=0.25)

    assert type(block.attention) is keras.layers.MultiHeadAttention
    assert block.attention.get_config()["dropout"] == 0.0
    assert isinstance(block.attention_dropout, keras.layers.Dropout)
    assert block.attention_dropout.rate == 0.25


@pytest.mark.parametrize("use_cache", [False, True])
@pytest.mark.parametrize("training", [False, True])
def test_output_dropout_is_used_by_regular_and_cached_calls(monkeypatch, x, use_cache, training):
    block = MultiHeadAttention(
        embed_dim=8,
        num_heads=2,
        dropout=0.25,
        gate_attention=False,
        gate_ffn=False,
    )
    block(x, x, training=False)

    calls = []
    original_call = block.attention_dropout.call

    def spy(inputs, training=False):
        calls.append(training)
        return original_call(inputs, training=training)

    monkeypatch.setattr(block.attention_dropout, "call", spy)

    if use_cache:
        cache = block.prepare_key_value(x)
        block.call_with_cache(x[:, :1], key_value_cache=cache, training=training)
    else:
        block(x, x, training=training)

    assert calls == [training]


@pytest.mark.parametrize("attention_dtype", [None, "float16", "bfloat16"])
def test_cached_call_matches_regular_call_without_dropout(x, attention_dtype):
    block = MultiHeadAttention(
        embed_dim=8,
        num_heads=2,
        dropout=0.0,
        flash_attention=False,
        attention_dtype=attention_dtype,
        gate_attention=False,
        gate_ffn=False,
    )

    expected = block(x[:, :1], x, training=False)
    cache = block.prepare_key_value(x)
    actual, returned_cache = block.call_with_cache(x[:, :1], key_value_cache=cache, training=False)

    np.testing.assert_allclose(
        keras.ops.convert_to_numpy(actual),
        keras.ops.convert_to_numpy(expected),
        rtol=1e-5,
        atol=1e-5,
    )
    assert returned_cache is cache


def test_flash_attention_false_overrides_global_auto_selection():
    previous = keras.config.is_flash_attention_enabled()
    keras.config.enable_flash_attention()
    try:
        block = MultiHeadAttention(embed_dim=8, num_heads=2, flash_attention=False)
        assert block.attention._flash_attention is False
    finally:
        if previous is False:
            keras.config.disable_flash_attention()
        else:
            keras.config.enable_flash_attention()


@pytest.mark.parametrize("attention_dtype", ["float16", "bfloat16"])
def test_attention_uses_mixed_policy_and_restores_output_dtype(x, attention_dtype):
    block = MultiHeadAttention(
        embed_dim=8,
        num_heads=2,
        dropout=0.0,
        flash_attention=False,
        attention_dtype=attention_dtype,
        gate_attention=False,
        gate_ffn=False,
    )

    output = block(x, x, training=False)
    cache = block.prepare_key_value(x)
    cached_output, _ = block.call_with_cache(x[:, :1], key_value_cache=cache, training=False)

    assert keras.backend.standardize_dtype(block.attention.compute_dtype) == attention_dtype
    assert keras.backend.standardize_dtype(block.attention.variable_dtype) == "float32"
    assert all(keras.backend.standardize_dtype(variable.dtype) == "float32" for variable in block.attention.weights)
    assert keras.backend.standardize_dtype(output.dtype) == keras.backend.standardize_dtype(x.dtype)
    assert keras.backend.standardize_dtype(cached_output.dtype) == keras.backend.standardize_dtype(x.dtype)


def test_invalid_attention_dtype_raises():
    with pytest.raises(ValueError, match="`attention_dtype` must be one of"):
        MultiHeadAttention(embed_dim=8, num_heads=2, attention_dtype="float32")


@pytest.mark.parametrize("flash_attention", [None, False, True])
def test_wrapper_config_round_trip(flash_attention):
    block = MultiHeadAttention(
        embed_dim=8,
        num_heads=2,
        dropout=0.1,
        flash_attention=flash_attention,
        attention_dtype="bfloat16",
    )

    config = block.get_config()
    reloaded = MultiHeadAttention.from_config(config)

    assert config["dropout"] == 0.1
    assert config["flash_attention"] is flash_attention
    assert config["attention_dtype"] == "bfloat16"
    assert reloaded.dropout_rate == 0.1
    assert reloaded.flash_attention is flash_attention
    assert reloaded.attention_dtype == "bfloat16"
    assert reloaded.attention.get_config()["dropout"] == 0.0
    assert reloaded.attention_dropout.rate == 0.1
    assert reloaded.attention.dtype_policy.name == "mixed_bfloat16"


def test_legacy_config_uses_new_option_defaults():
    config = MultiHeadAttention(embed_dim=8, num_heads=2).get_config()
    config.pop("flash_attention")
    config.pop("attention_dtype")

    reloaded = MultiHeadAttention.from_config(config)

    assert reloaded.flash_attention is None
    assert reloaded.attention_dtype is None


def _attention_blocks(network):
    return [layer for layer in network._flatten_layers() if isinstance(layer, MultiHeadAttention)]


@pytest.mark.parametrize(
    "network_name, kwargs",
    [
        ("SetTransformer", dict(embed_dims=(8, 8), num_heads=(2, 2))),
        ("SetTransformer", dict(embed_dims=(8, 8), num_heads=(2, 2), num_inducing_points=3)),
        ("TimeSeriesTransformer", dict(embed_dims=(8, 8), num_heads=(2, 2))),
        ("FusionTransformer", dict(embed_dims=(8, 8), num_heads=(2, 2), template_dim=8)),
    ],
)
def test_options_propagate_and_serialize(network_name, kwargs, x):
    import bayesflow.networks

    network = getattr(bayesflow.networks, network_name)(
        summary_dim=4,
        dropout=0.2,
        flash_attention=False,
        attention_dtype="bfloat16",
        **kwargs,
    )
    network(x, training=False)
    reloaded = deserialize(serialize(network))
    reloaded(x, training=False)

    for net in (network, reloaded):
        blocks = _attention_blocks(net)
        assert blocks
        assert all(type(block.attention) is keras.layers.MultiHeadAttention for block in blocks)
        assert all(block.attention.get_config()["dropout"] == 0.0 for block in blocks)
        assert all(block.attention_dropout.rate == 0.2 for block in blocks)
        assert all(block.flash_attention is False for block in blocks)
        assert all(block.attention._flash_attention is False for block in blocks)
        assert all(block.attention_dtype == "bfloat16" for block in blocks)
        assert all(block.attention.dtype_policy.name == "mixed_bfloat16" for block in blocks)
