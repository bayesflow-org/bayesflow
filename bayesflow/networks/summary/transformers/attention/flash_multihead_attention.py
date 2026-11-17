import keras

from bayesflow.utils.serialization import serializable


@serializable("bayesflow.networks")
class FlashMultiHeadAttention(keras.layers.MultiHeadAttention):
    """Multi-head attention that can use FlashAttention at inference despite training dropout.

    Keras only uses :func:`keras.ops.dot_product_attention`, which can dispatch to
    FlashAttention, when the layer's dropout rate is zero. Configuring dropout for training
    therefore also disables FlashAttention during inference, even though dropout is inactive.

    This layer keeps standard dropout-enabled attention during training and uses FlashAttention
    whenever dropout is inactive and FlashAttention is available (e.g., on a GPU with ``float16``
    or ``bfloat16`` inputs). In all other cases, it behaves exactly like
    ``keras.layers.MultiHeadAttention``. In particular, it never routes CPU calls through the
    generic fused kernel, which is considerably slower than the standard path on CPU.

    Notes
    -----
    The implementation overrides Keras's ``_compute_attention`` method and accesses the private
    attributes ``_dropout``, ``_attention_axes`` and ``_inverse_sqrt_key_dim``. Compatibility is
    therefore covered by dedicated tests.

    Parameters
    ----------
    dropout : float, optional
        Dropout applied to attention scores during training, by default 0.0.
    flash_attention : bool or None, optional
        FlashAttention policy. If None, FlashAttention is used whenever it is available, unless
        it has been disabled via ``keras.config.disable_flash_attention()``. If True,
        FlashAttention is required and unsupported inputs raise an error. If False, the layer
        behaves exactly like ``keras.layers.MultiHeadAttention``. By default None.
    attention_dtype : {"float16", "bfloat16"} or None, optional
        Dtype of query, key and value when FlashAttention is used. The attention output is cast
        back to the layer's compute dtype, so reduced-precision attention does not require
        changing the dtype policy of the rest of the model. FlashAttention kernels typically
        require one of these dtypes. If None, no cast is performed. By default None.
    **kwargs
        Additional keyword arguments passed to ``keras.layers.MultiHeadAttention``.
    """

    def __init__(
        self,
        *args,
        dropout: float = 0.0,
        flash_attention: bool | None = None,
        attention_dtype: str | None = None,
        **kwargs,
    ):
        if attention_dtype not in (None, "float16", "bfloat16"):
            raise ValueError(
                "`attention_dtype` must be one of None, 'float16', or 'bfloat16'. "
                f"Received: attention_dtype={attention_dtype!r}."
            )

        # Keras rejects dropout combined with flash attention, so initialize with a valid
        # combination and restore the requested dropout before the layer is built.
        super().__init__(*args, dropout=0.0, flash_attention=False, **kwargs)
        self._dropout = dropout
        self._flash_attention = False
        self._fast_path_flash_attention = flash_attention
        self._attention_dtype = attention_dtype

    def _flash_attention_supported(self, query) -> bool:
        """Whether FlashAttention can run for ``query`` on the current backend and device."""
        dtype = self._attention_dtype or keras.backend.standardize_dtype(query.dtype)
        if dtype not in ("float16", "bfloat16"):
            return False

        match keras.backend.backend():
            case "jax":
                import jax

                return jax.default_backend() in ("gpu", "tpu")
            case "torch":
                return query.is_cuda
            case _:
                return False

    def _resolve_flash_attention(self, query) -> bool | None:
        """Returns False for the standard path, otherwise the policy passed to the fused kernel."""
        flash_attention = self._fast_path_flash_attention
        if flash_attention is not None:
            return flash_attention

        if keras.config.is_flash_attention_enabled() is False:
            return False

        # None lets the backend pick its FlashAttention kernel and fall back if inputs do not qualify
        return None if self._flash_attention_supported(query) else False

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
        flash_attention = self._resolve_flash_attention(query)
        use_fast_path = flash_attention is not False and not (
            (self._dropout > 0.0 and training) or return_attention_scores or len(query.shape) != 4
        )
        if not use_fast_path:
            return super()._compute_attention(
                query,
                key,
                value,
                attention_mask=attention_mask,
                training=training,
                return_attention_scores=return_attention_scores,
                use_causal_mask=use_causal_mask,
            )

        if attention_mask is not None:
            # expand (B, T, S) masks for broadcasting over attention heads
            mask_expansion_axis = -len(self._attention_axes) * 2 - 1
            for _ in range(4 - len(attention_mask.shape)):
                attention_mask = keras.ops.expand_dims(attention_mask, axis=mask_expansion_axis)
            attention_mask = keras.ops.cast(attention_mask, "bool")

        output_dtype = query.dtype
        if self._attention_dtype is not None:
            query = keras.ops.cast(query, self._attention_dtype)
            key = keras.ops.cast(key, self._attention_dtype)
            value = keras.ops.cast(value, self._attention_dtype)

        attention_output = keras.ops.dot_product_attention(
            query=query,
            key=key,
            value=value,
            bias=None,
            mask=attention_mask,
            scale=self._inverse_sqrt_key_dim,
            is_causal=use_causal_mask,
            flash_attention=flash_attention,
        )
        if self._attention_dtype is not None:
            attention_output = keras.ops.cast(attention_output, output_dtype)
        return attention_output, None

    def get_config(self) -> dict:
        base_config = super().get_config()
        return base_config | {
            "flash_attention": self._fast_path_flash_attention,
            "attention_dtype": self._attention_dtype,
        }
