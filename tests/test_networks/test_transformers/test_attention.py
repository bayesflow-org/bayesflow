import keras
import numpy as np
import pytest

from bayesflow.networks import SetTransformer
from bayesflow.networks.transformers.attention import FlashMultiHeadAttention
from bayesflow.networks.transformers.isab import InducedSetAttentionBlock
from bayesflow.networks.transformers.mab import MultiHeadAttentionBlock
from bayesflow.networks.transformers.pma import PoolingByMultiHeadAttention
from bayesflow.utils.serialization import deserialize, serialize


@pytest.fixture()
def x():
    return keras.random.normal((2, 6, 3), seed=0)


def test_relied_upon_keras_internals_are_present():
    # Fail clearly if a Keras update removes private attributes used by
    # `FlashMultiHeadAttention`; see the compatibility note on that class.
    attn = FlashMultiHeadAttention(key_dim=8, num_heads=2, dropout=0.1)
    attn.build((2, 6, 3), (2, 6, 3))

    assert isinstance(attn._dropout, float)
    assert attn._dropout == 0.1
    assert isinstance(attn._attention_axes, tuple) and len(attn._attention_axes) > 0
    assert isinstance(attn._inverse_sqrt_key_dim, float)
    assert callable(getattr(keras.layers.MultiHeadAttention, "_compute_attention", None))


def spy_on_dot_product_attention(monkeypatch):
    """Track calls to the fused dot-product attention operation used by `attention.py`."""
    import bayesflow.networks.transformers.attention as attention_module

    calls = {"n": 0, "kwargs": None}
    original = attention_module.ops.dot_product_attention

    def spy(*args, **kwargs):
        calls["n"] += 1
        calls["kwargs"] = kwargs
        return original(*args, **kwargs)

    monkeypatch.setattr(attention_module.ops, "dot_product_attention", spy)
    return calls


class TestFastPathGating:
    def test_training_with_dropout_never_uses_fast_path(self, monkeypatch, x):
        block = MultiHeadAttentionBlock(embed_dim=8, num_heads=2, dropout=0.3, layer_norm=False)
        block(x, x, training=False)  # build
        calls = spy_on_dot_product_attention(monkeypatch)

        block(x, x, training=True)
        assert calls["n"] == 0

    def test_inference_with_nonzero_configured_dropout_uses_fast_path(self, monkeypatch, x):
        block = MultiHeadAttentionBlock(embed_dim=8, num_heads=2, dropout=0.3, layer_norm=False)
        block(x, x, training=False)  # build
        calls = spy_on_dot_product_attention(monkeypatch)

        block(x, x, training=False)
        assert calls["n"] == 1

    def test_return_attention_scores_never_uses_fast_path(self, monkeypatch, x):
        attn = FlashMultiHeadAttention(key_dim=8, num_heads=2, dropout=0.0)
        attn(query=x, key=x, value=x, training=False)  # build
        calls = spy_on_dot_product_attention(monkeypatch)

        output, scores = attn(query=x, key=x, value=x, training=False, return_attention_scores=True)
        assert calls["n"] == 0
        assert scores is not None
        assert keras.ops.shape(output)[:-1] == keras.ops.shape(x)[:-1]

    def test_flash_attention_false_is_forwarded_on_fast_path(self, monkeypatch, x):
        attn = FlashMultiHeadAttention(key_dim=8, num_heads=2, dropout=0.0, flash_attention=False)
        attn(query=x, key=x, value=x, training=False)  # build
        calls = spy_on_dot_product_attention(monkeypatch)

        attn(query=x, key=x, value=x, training=False)
        assert calls["n"] == 1
        assert calls["kwargs"]["flash_attention"] is False

    def test_default_flash_attention_respects_global_disabling(self, monkeypatch, x):
        # Regression test: `flash_attention=None` (the default) must defer to
        # `keras.config.disable_flash_attention()` rather than always attempting flash attention
        # opportunistically, since `ops.dot_product_attention` never consults that global setting
        # on its own -- only `keras.layers.MultiHeadAttention.__init__` does, which this class
        # bypasses.
        attn = FlashMultiHeadAttention(key_dim=8, num_heads=2, dropout=0.0)  # flash_attention=None
        attn(query=x, key=x, value=x, training=False)  # build
        calls = spy_on_dot_product_attention(monkeypatch)

        was_disabled = keras.config.is_flash_attention_enabled() is False
        keras.config.disable_flash_attention()
        try:
            attn(query=x, key=x, value=x, training=False)
        finally:
            if not was_disabled:
                keras.config.enable_flash_attention()

        assert calls["n"] == 1
        assert calls["kwargs"]["flash_attention"] is False

    def test_construction_safe_under_global_flash_attention_enabled(self, x):
        # Global FlashAttention must not make construction fail when training dropout is nonzero.
        was_disabled = keras.config.is_flash_attention_enabled() is False
        keras.config.enable_flash_attention()
        try:
            block = MultiHeadAttentionBlock(embed_dim=8, num_heads=2, dropout=0.05, layer_norm=False)
            block(x, x, training=True)
            block(x, x, training=False)
        finally:
            # Restore the global setting to avoid affecting later tests.
            if was_disabled:
                keras.config.disable_flash_attention()
            else:
                keras.config.enable_flash_attention()


def test_fast_path_matches_manual_path_at_dropout_zero(x):
    attn = FlashMultiHeadAttention(key_dim=8, num_heads=2, dropout=0.0)
    fast_output = attn(query=x, key=x, value=x, training=False)  # build + fast path

    query = attn._query_dense(x)
    key = attn._key_dense(x)
    value = attn._value_dense(x)
    # Requesting scores forces Keras's explicit softmax/einsum reference path. Without it, both
    # sides of this comparison would use fused dot-product attention.
    manual_output, _ = keras.layers.MultiHeadAttention._compute_attention(attn, query, key, value, None, False, True)
    manual_output = attn._output_dense(manual_output)

    np.testing.assert_allclose(
        keras.ops.convert_to_numpy(fast_output), keras.ops.convert_to_numpy(manual_output), atol=1e-5
    )


def test_attention_mask_matches_manual_path(x):
    batch, seq = keras.ops.shape(x)[0], keras.ops.shape(x)[1]
    mask_np = np.ones((batch, seq, seq), dtype=bool)
    mask_np[:, :, -1] = False  # mask out attending to the last key position
    mask = keras.ops.convert_to_tensor(mask_np)

    attn = FlashMultiHeadAttention(key_dim=8, num_heads=2, dropout=0.0)
    fast_output = attn(query=x, key=x, value=x, training=False, attention_mask=mask)
    assert np.all(np.isfinite(keras.ops.convert_to_numpy(fast_output)))

    query = attn._query_dense(x)
    key = attn._key_dense(x)
    value = attn._value_dense(x)
    manual_output, _ = keras.layers.MultiHeadAttention._compute_attention(attn, query, key, value, mask, False, True)
    manual_output = attn._output_dense(manual_output)

    np.testing.assert_allclose(
        keras.ops.convert_to_numpy(fast_output), keras.ops.convert_to_numpy(manual_output), atol=1e-5
    )


@pytest.mark.numpy
@pytest.mark.tensorflow
def test_flash_attention_true_raises_on_unsupported_backend(x):
    # NumPy and TensorFlow reject forced FlashAttention. JAX and PyTorch eligibility depends on
    # hardware, so their behavior cannot be asserted in this backend-independent test suite.
    attn = FlashMultiHeadAttention(key_dim=8, num_heads=2, dropout=0.0, flash_attention=True)
    with pytest.raises(ValueError, match="(?i)flash attention"):
        attn(query=x, key=x, value=x, training=False)


class TestFlashAttentionPropagation:
    def test_isab_propagates_to_both_mab_blocks(self):
        isab = InducedSetAttentionBlock(num_inducing_points=4, embed_dim=8, num_heads=2, flash_attention=False)
        assert isab.mab0.attention._fast_path_flash_attention is False
        assert isab.mab1.attention._fast_path_flash_attention is False

    def test_pma_propagates_to_its_mab(self):
        pma = PoolingByMultiHeadAttention(embed_dim=8, num_heads=2, flash_attention=False)
        assert pma.mab.attention._fast_path_flash_attention is False

    def test_set_transformer_propagates_flash_attention_setting(self, x):
        st = SetTransformer(
            summary_dim=4, embed_dims=(8, 8), num_heads=(2, 2), num_inducing_points=3, flash_attention=False
        )
        st(x, training=False)  # build

        for block in st.attention_blocks.layers:
            assert block.mab0.attention._fast_path_flash_attention is False
            assert block.mab1.attention._fast_path_flash_attention is False
        assert st.pooling_by_attention.mab.attention._fast_path_flash_attention is False


class TestSetTransformerISAB:
    @pytest.mark.parametrize("training", [True, False])
    def test_builds_and_runs(self, x, training):
        st = SetTransformer(summary_dim=4, embed_dims=(8, 8), num_heads=(2, 2), num_inducing_points=3)
        output = st(x, training=training)
        assert keras.ops.shape(output) == (keras.ops.shape(x)[0], 4)


class TestSerialization:
    def test_flash_multihead_attention_round_trip(self, x):
        attn = FlashMultiHeadAttention(key_dim=8, num_heads=2, dropout=0.2, flash_attention=False)
        attn(query=x, key=x, value=x, training=False)  # build

        config = serialize(attn)
        reloaded = deserialize(config)

        assert reloaded._dropout == 0.2
        assert reloaded._fast_path_flash_attention is False
        assert keras.tree.lists_to_tuples(config) == keras.tree.lists_to_tuples(serialize(reloaded))

    def test_set_transformer_isab_flash_attention_round_trip(self, x):
        st = SetTransformer(
            summary_dim=4,
            embed_dims=(8,),
            num_heads=(2,),
            mlp_depths=(2,),
            mlp_widths=(32,),
            num_inducing_points=3,
            flash_attention=False,
        )
        st(x, training=False)

        config = serialize(st)
        reloaded = deserialize(config)
        reloaded(x, training=False)

        for block in reloaded.attention_blocks.layers:
            assert block.mab0.attention._fast_path_flash_attention is False
            assert block.mab1.attention._fast_path_flash_attention is False
