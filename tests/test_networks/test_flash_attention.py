"""Tests for the fused attention path of ``FlashMultiHeadAttention`` and its propagation."""

import keras
import numpy as np
import pytest

from bayesflow.networks.summary.transformers.attention import FlashMultiHeadAttention, MultiHeadAttention
from bayesflow.utils.serialization import deserialize, serialize


@pytest.fixture()
def x():
    return keras.random.normal((2, 6, 3), seed=0)


def spy_on_dot_product_attention(monkeypatch):
    """Track calls to the fused path. Keras's own fallback imports ``ops`` internally and is not counted."""
    calls = {"n": 0, "kwargs": None}
    original = keras.ops.dot_product_attention

    def spy(*args, **kwargs):
        calls["n"] += 1
        calls["kwargs"] = kwargs
        return original(*args, **kwargs)

    monkeypatch.setattr(keras.ops, "dot_product_attention", spy)
    return calls


def test_relied_upon_keras_internals_are_present():
    # fail clearly if a Keras update removes private attributes used by FlashMultiHeadAttention
    attn = FlashMultiHeadAttention(key_dim=8, num_heads=2, dropout=0.1)
    attn.build((2, 6, 3), (2, 6, 3))

    assert attn._dropout == 0.1
    assert isinstance(attn._attention_axes, tuple) and len(attn._attention_axes) > 0
    assert isinstance(attn._inverse_sqrt_key_dim, float)
    assert callable(getattr(keras.layers.MultiHeadAttention, "_compute_attention", None))


@pytest.fixture()
def flash_supported(monkeypatch):
    """Pretend FlashAttention is available, so the fused path runs on CPU with the backend's generic kernel."""
    monkeypatch.setattr(FlashMultiHeadAttention, "_flash_attention_supported", lambda self, query: True)


class TestFastPathGating:
    def test_training_with_dropout_never_uses_fast_path(self, monkeypatch, flash_supported, x):
        block = MultiHeadAttention(embed_dim=8, num_heads=2, dropout=0.3, layer_norm=False)
        block(x, x, training=False)
        calls = spy_on_dot_product_attention(monkeypatch)

        block(x, x, training=True)
        assert calls["n"] == 0

    def test_inference_with_nonzero_configured_dropout_uses_fast_path(self, monkeypatch, flash_supported, x):
        block = MultiHeadAttention(embed_dim=8, num_heads=2, dropout=0.3, layer_norm=False)
        block(x, x, training=False)
        calls = spy_on_dot_product_attention(monkeypatch)

        block(x, x, training=False)
        assert calls["n"] == 1
        assert calls["kwargs"]["flash_attention"] is None

    def test_training_without_dropout_uses_fast_path(self, monkeypatch, flash_supported, x):
        block = MultiHeadAttention(embed_dim=8, num_heads=2, dropout=0.0, layer_norm=False)
        block(x, x, training=False)
        calls = spy_on_dot_product_attention(monkeypatch)

        block(x, x, training=True)
        assert calls["n"] == 1

    def test_return_attention_scores_never_uses_fast_path(self, monkeypatch, flash_supported, x):
        attn = FlashMultiHeadAttention(key_dim=8, num_heads=2)
        attn(query=x, key=x, value=x, training=False)
        calls = spy_on_dot_product_attention(monkeypatch)

        output, scores = attn(query=x, key=x, value=x, training=False, return_attention_scores=True)
        assert calls["n"] == 0
        assert scores is not None
        assert keras.ops.shape(output)[:-1] == keras.ops.shape(x)[:-1]

    def test_unsupported_hardware_uses_standard_path(self, monkeypatch, x):
        # the generic fused kernel is much slower than the standard path on CPU, so it is never used there
        monkeypatch.setattr(FlashMultiHeadAttention, "_flash_attention_supported", lambda self, query: False)
        attn = FlashMultiHeadAttention(key_dim=8, num_heads=2, dropout=0.3)
        attn(query=x, key=x, value=x, training=False)
        calls = spy_on_dot_product_attention(monkeypatch)

        attn(query=x, key=x, value=x, training=False)
        assert calls["n"] == 0

    def test_flash_attention_false_uses_standard_path(self, monkeypatch, flash_supported, x):
        attn = FlashMultiHeadAttention(key_dim=8, num_heads=2, flash_attention=False)
        attn(query=x, key=x, value=x, training=False)
        calls = spy_on_dot_product_attention(monkeypatch)

        attn(query=x, key=x, value=x, training=False)
        assert calls["n"] == 0

    def test_default_flash_attention_respects_global_disabling(self, monkeypatch, flash_supported, x):
        attn = FlashMultiHeadAttention(key_dim=8, num_heads=2)
        attn(query=x, key=x, value=x, training=False)
        calls = spy_on_dot_product_attention(monkeypatch)

        previous = keras.config.is_flash_attention_enabled()
        keras.config.disable_flash_attention()
        try:
            attn(query=x, key=x, value=x, training=False)
        finally:
            if previous is not False:
                keras.config.enable_flash_attention()

        assert calls["n"] == 0

    def test_construction_safe_under_global_flash_attention_enabled(self, x):
        previous = keras.config.is_flash_attention_enabled()
        keras.config.enable_flash_attention()
        try:
            block = MultiHeadAttention(embed_dim=8, num_heads=2, dropout=0.05, layer_norm=False)
            block(x, x, training=True)
        finally:
            if previous is False:
                keras.config.disable_flash_attention()

    def test_call_with_cache_uses_fast_path(self, monkeypatch, flash_supported, x):
        block = MultiHeadAttention(embed_dim=8, num_heads=2, dropout=0.3, layer_norm=False)
        block(x, x, training=False)
        calls = spy_on_dot_product_attention(monkeypatch)

        cache = block.prepare_key_value(x)
        block.call_with_cache(x[:, :1], key_value_cache=cache)
        assert calls["n"] == 1


class TestFlashAttentionSupport:
    def test_float32_is_not_supported(self, x):
        attn = FlashMultiHeadAttention(key_dim=8, num_heads=2)
        assert not attn._flash_attention_supported(x)

    @pytest.mark.jax
    def test_reduced_precision_is_not_supported_on_jax_cpu(self, x):
        import jax

        if jax.default_backend() != "cpu":
            pytest.skip("requires a CPU-only JAX installation")
        attn = FlashMultiHeadAttention(key_dim=8, num_heads=2, attention_dtype="bfloat16")
        assert not attn._flash_attention_supported(x)


class TestAttentionDtype:
    @pytest.mark.parametrize("attention_dtype", ["float16", "bfloat16"])
    def test_casts_qkv_and_restores_output_dtype(self, monkeypatch, flash_supported, x, attention_dtype):
        attn = FlashMultiHeadAttention(key_dim=8, num_heads=2, attention_dtype=attention_dtype)
        calls = spy_on_dot_product_attention(monkeypatch)

        output = attn(query=x, key=x, value=x, training=False)

        assert calls["n"] == 1
        for name in ("query", "key", "value"):
            assert keras.backend.standardize_dtype(calls["kwargs"][name].dtype) == attention_dtype
        assert output.dtype == x.dtype

    def test_default_does_not_cast(self, monkeypatch, flash_supported, x):
        attn = FlashMultiHeadAttention(key_dim=8, num_heads=2)
        calls = spy_on_dot_product_attention(monkeypatch)

        output = attn(query=x, key=x, value=x, training=False)

        for name in ("query", "key", "value"):
            assert calls["kwargs"][name].dtype == x.dtype
        assert output.dtype == x.dtype

    @pytest.mark.parametrize("attention_dtype", ["float16", "bfloat16"])
    def test_reduced_precision_is_close_to_float32(self, flash_supported, x, attention_dtype):
        reference = FlashMultiHeadAttention(key_dim=8, num_heads=2, flash_attention=False)
        reduced = FlashMultiHeadAttention(key_dim=8, num_heads=2, attention_dtype=attention_dtype)
        reference(query=x, key=x, value=x)
        reduced(query=x, key=x, value=x)
        reduced.set_weights(reference.get_weights())

        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(reduced(query=x, key=x, value=x)),
            keras.ops.convert_to_numpy(reference(query=x, key=x, value=x)),
            atol=5e-2,
        )

    def test_invalid_value_raises(self):
        with pytest.raises(ValueError, match="`attention_dtype` must be one of"):
            FlashMultiHeadAttention(key_dim=8, num_heads=2, attention_dtype="float32")


def _manual_attention(attn, x, mask=None):
    query = attn._query_dense(x)
    key = attn._key_dense(x)
    value = attn._value_dense(x)
    # requesting scores forces Keras's explicit softmax/einsum reference path
    output, _ = keras.layers.MultiHeadAttention._compute_attention(attn, query, key, value, mask, False, True)
    return attn._output_dense(output)


def test_fast_path_matches_reference_path(flash_supported, x):
    attn = FlashMultiHeadAttention(key_dim=8, num_heads=2)
    fast_output = attn(query=x, key=x, value=x, training=False)

    np.testing.assert_allclose(
        keras.ops.convert_to_numpy(fast_output), keras.ops.convert_to_numpy(_manual_attention(attn, x)), atol=1e-5
    )


def test_attention_mask_matches_reference_path(flash_supported, x):
    mask_np = np.ones((2, 6, 6), dtype=bool)
    mask_np[:, :, -1] = False
    mask = keras.ops.convert_to_tensor(mask_np)

    attn = FlashMultiHeadAttention(key_dim=8, num_heads=2)
    fast_output = attn(query=x, key=x, value=x, training=False, attention_mask=mask)

    np.testing.assert_allclose(
        keras.ops.convert_to_numpy(fast_output),
        keras.ops.convert_to_numpy(_manual_attention(attn, x, mask)),
        atol=1e-5,
    )


def test_causal_mask_matches_reference_path(flash_supported, x):
    attn = FlashMultiHeadAttention(key_dim=8, num_heads=2)
    fast_output = attn(query=x, key=x, value=x, training=False, use_causal_mask=True)
    causal_mask = keras.ops.convert_to_tensor(np.tril(np.ones((2, 6, 6), dtype=bool)))

    np.testing.assert_allclose(
        keras.ops.convert_to_numpy(fast_output),
        keras.ops.convert_to_numpy(_manual_attention(attn, x, causal_mask)),
        atol=1e-5,
    )


@pytest.mark.numpy
@pytest.mark.tensorflow
def test_flash_attention_true_raises_on_unsupported_backend(x):
    # JAX and PyTorch eligibility depends on hardware, so only these backends are asserted
    attn = FlashMultiHeadAttention(key_dim=8, num_heads=2, flash_attention=True)
    with pytest.raises(ValueError, match="(?i)flash attention"):
        attn(query=x, key=x, value=x, training=False)


def _attention_layers(network):
    return [layer for layer in network._flatten_layers() if isinstance(layer, FlashMultiHeadAttention)]


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
        summary_dim=4, flash_attention=False, attention_dtype="bfloat16", **kwargs
    )
    network(x, training=False)
    reloaded = deserialize(serialize(network))
    reloaded(x, training=False)

    for net in (network, reloaded):
        layers = _attention_layers(net)
        assert layers
        assert all(layer._fast_path_flash_attention is False for layer in layers)
        assert all(layer._attention_dtype == "bfloat16" for layer in layers)
