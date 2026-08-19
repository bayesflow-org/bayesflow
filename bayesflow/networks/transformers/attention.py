import inspect

import keras
import keras.ops as ops

from bayesflow.utils.serialization import serializable

# Keras 3.12 folds causal masking into `attention_mask` before calling
# `_compute_attention`, while newer versions pass `use_causal_mask` explicitly.
# Inspect the method once so the fallback works with both signatures.
_BASE_COMPUTE_ATTENTION_ACCEPTS_USE_CAUSAL_MASK = (
    "use_causal_mask" in inspect.signature(keras.layers.MultiHeadAttention._compute_attention).parameters
)


@serializable("bayesflow.networks")
class FlashMultiHeadAttention(keras.layers.MultiHeadAttention):
    """Multi-head attention with a fused inference path independent of training dropout.

    Keras normally uses :func:`keras.ops.dot_product_attention`, which can dispatch to
    FlashAttention, only when the layer's configured dropout rate is zero. Consequently,
    configuring dropout for training also disables fused attention during inference, even though
    dropout is inactive then.

    This layer preserves standard dropout-enabled attention during training and uses fused
    dot-product attention during inference. When ``dropout == 0``, the fused path is also available
    during training. Calls that request attention scores continue to use the standard path.

    Notes
    -----
    The implementation overrides Keras's ``_compute_attention`` method and accesses the private
    attributes ``_dropout``, ``_attention_axes``, and ``_inverse_sqrt_key_dim``. Compatibility is
    therefore covered by dedicated tests. Keras 3.12 and newer releases also use different
    ``_compute_attention`` signatures for causal masking; the module-level signature check handles
    both forms.
    """

    def __init__(self, *args, dropout: float = 0.0, flash_attention: bool | None = None, **kwargs):
        """Create a multi-head attention layer with an inference-time fused path.

        Parameters
        ----------
        dropout : float, optional (default - 0.0)
            Dropout applied to attention scores during training.
        flash_attention : bool or None, optional (default - None)
            FlashAttention policy for eligible fused calls. ``None`` defers to
            ``keras.config.is_flash_attention_enabled()`` when the attention computation is traced
            or executed; ``True`` requires it; ``False`` disables it while retaining fused
            dot-product attention. This setting does not affect training calls with nonzero dropout
            or calls that request attention scores.
        *args, **kwargs
            Additional arguments forwarded to :class:`keras.layers.MultiHeadAttention`.
        """
        # Initialize Keras with a valid dropout/FlashAttention combination, then restore the
        # requested dropout before the layer is built. This class manages FlashAttention for its
        # fused path separately through `_fast_path_flash_attention`.
        super().__init__(*args, dropout=0.0, flash_attention=False, **kwargs)
        self._dropout = dropout
        self._flash_attention = False
        self._fast_path_flash_attention = flash_attention

    def _compute_attention(
        self,
        query,
        key,
        value,
        attention_mask=None,
        training=None,
        return_attention_scores=False,
        use_causal_mask=False,
    ):
        use_fast_path = not ((self._dropout > 0.0 and training) or return_attention_scores or (len(query.shape) != 4))
        if not use_fast_path:
            fallback_kwargs = dict(
                attention_mask=attention_mask,
                training=training,
                return_attention_scores=return_attention_scores,
            )
            if _BASE_COMPUTE_ATTENTION_ACCEPTS_USE_CAUSAL_MASK:
                fallback_kwargs["use_causal_mask"] = use_causal_mask
            return super()._compute_attention(query, key, value, **fallback_kwargs)

        if attention_mask is not None:
            # Expand `(B, T, S)` masks for broadcasting over attention heads.
            mask_expansion_axis = -len(self._attention_axes) * 2 - 1
            for _ in range(4 - len(attention_mask.shape)):
                attention_mask = ops.expand_dims(attention_mask, axis=mask_expansion_axis)
            attention_mask = ops.cast(attention_mask, dtype="bool")

        flash_attention = self._fast_path_flash_attention
        if flash_attention is None:
            # `ops.dot_product_attention` doesn't consult the global flash-attention setting
            # itself, so resolve it here instead of leaving `flash_attention=None`.
            flash_attention = keras.config.is_flash_attention_enabled()

        attention_output = ops.dot_product_attention(
            query=query,
            key=key,
            value=value,
            bias=None,
            mask=attention_mask,
            scale=self._inverse_sqrt_key_dim,
            is_causal=use_causal_mask,
            flash_attention=flash_attention,
        )
        return attention_output, None

    def get_config(self):
        config = {"flash_attention": self._fast_path_flash_attention}
        return {**super().get_config(), **config}
