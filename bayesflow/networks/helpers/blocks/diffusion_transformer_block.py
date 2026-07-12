import keras
from keras import layers

from bayesflow.types import Tensor
from bayesflow.utils import layer_kwargs
from bayesflow.utils.serialization import deserialize, serializable, serialize

from .transformer_feedforward import FFN


@serializable("bayesflow.networks")
class QKNormMultiHeadAttention(layers.MultiHeadAttention):
    """Multi-head attention with RMS-normalized queries and keys (QK-norm)."""

    def build(self, query_shape, value_shape, key_shape=None):
        super().build(query_shape, value_shape, key_shape)
        head_shape = (None, None, self._num_heads, self._key_dim)
        self.query_norm = keras.layers.RMSNormalization(axis=-1)
        self.query_norm.build(head_shape)
        self.key_norm = keras.layers.RMSNormalization(axis=-1)
        self.key_norm.build(head_shape)

    def _compute_attention(self, query, key, value, *args, **kwargs):
        return super()._compute_attention(self.query_norm(query), self.key_norm(key), value, *args, **kwargs)


@serializable("bayesflow.networks")
class DiffusionTransformerBlock(keras.Layer):
    """Transformer block with adaLN-single time conditioning and QK-normalized attention.

    Target tokens are the residual stream; optional condition tokens enter only as
    frozen keys/values (cross-attention style). Queries and keys are RMS-normalized
    per head (QK-norm) before the attention dot product, which stabilizes training
    regardless of the scale of the query/key projections.

    Parameters
    ----------
    width : int
        Token embedding width.
    num_heads : int, optional
        Number of attention heads. Default is ``4``.
    dropout : float, optional
        Dropout rate used in attention and feedforward sublayers. Default is
        ``0.0``.
    expansion_factor : float, optional
        Feedforward expansion factor. Default is ``4.0``.
    glu_variant : str, optional
        Gated activation variant for the feedforward network. One of
        ``"swiglu"``, ``"geglu"``, ``"reglu"``, or ``"liglu"``. Default is
        ``"swiglu"``.
    use_bias : bool, optional
        Whether dense projections include a bias term. Default is ``False``.
    residual_gate_init : float, optional
        Initial value for the adaLN residual gates. Default is ``1e-2``.
    kernel_initializer : str or keras.Initializer, optional
        Initializer for dense projection kernels. Default is
        ``"glorot_uniform"``.
    **kwargs
        Additional keyword arguments forwarded to ``keras.Layer``.
    """

    def __init__(
        self,
        width: int,
        *,
        num_heads: int = 4,
        dropout: float = 0.0,
        expansion_factor: float = 4.0,
        glu_variant: str = "swiglu",
        use_bias: bool = False,
        residual_gate_init: float = 1e-2,
        kernel_initializer: str | keras.Initializer = "glorot_uniform",
        **kwargs,
    ):
        super().__init__(**layer_kwargs(kwargs))

        if width % num_heads != 0:
            raise ValueError("TimeTransformerBlock requires width to be divisible by num_heads.")

        self.width = width
        self.num_heads = num_heads
        self.dropout = dropout
        self.expansion_factor = expansion_factor
        self.glu_variant = glu_variant
        self.use_bias = use_bias
        self.residual_gate_init = residual_gate_init
        self.kernel_initializer = kernel_initializer

        # adaLN supplies the affine part, so the norms are non-affine (DiT-style).
        self.attn_norm = keras.layers.LayerNormalization(center=False, scale=False)
        self.ffn_norm = keras.layers.LayerNormalization(center=False, scale=False)

        # MHA-internal dropout is 0, so regularize the attention output instead of the probs (faster).
        self.attn = QKNormMultiHeadAttention(
            key_dim=width // num_heads,
            num_heads=num_heads,
            dropout=0.0,
            use_bias=use_bias,
            output_shape=width,
            kernel_initializer=kernel_initializer,
        )
        self.attn_dropout = keras.layers.Dropout(dropout)

        self.ffn = FFN(
            embed_dim=width,
            expansion_factor=expansion_factor,
            glu_variant=glu_variant,
            dropout=dropout,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
        )

        # adaLN-single: the (shift, scale, gate) modulation for the attention and FFN sublayers.
        self.ada_ln_table = None

    def build(self, input_shape):
        if self.built:
            return

        x_shape, _ = input_shape
        self.attn_norm.build(x_shape)
        self.ffn_norm.build(x_shape)
        self.attn.build(query_shape=x_shape, value_shape=x_shape)

        self.ffn.build(x_shape)

        self.ada_ln_table = self.add_weight(shape=(6 * self.width,), initializer="zeros", name="ada_ln_table")
        if self.residual_gate_init != 0.0:
            self.ada_ln_table.assign(self.ada_ln_bias(self.width, self.residual_gate_init))

    def compute_output_shape(self, input_shape):
        return tuple(input_shape[0])

    def call(
        self,
        inputs: tuple[Tensor, Tensor],
        *,
        conditions: Tensor | None = None,
        attention_mask: Tensor | None = None,
        update_mask: Tensor | None = None,
        training: bool | None = None,
    ) -> Tensor:
        x, base_mod = inputs
        residual = x

        # adaLN-single: shared modulation from the network-level MLP plus this block's offset.
        mod = base_mod + self.ada_ln_table[None]
        mod = keras.ops.expand_dims(mod, axis=1)
        shift_attn, scale_attn, gate_attn, shift_ffn, scale_ffn, gate_ffn = keras.ops.split(mod, 6, axis=-1)

        attn_in = self.modulate(self.attn_norm(x, training=training), shift_attn, scale_attn)
        kv = attn_in if conditions is None else keras.ops.concatenate([attn_in, conditions], axis=1)
        attn_out = self.attn(attn_in, kv, attention_mask=attention_mask, training=training)
        h = x + gate_attn * self.attn_dropout(attn_out, training=training)

        ffn_in = self.modulate(self.ffn_norm(h, training=training), shift_ffn, scale_ffn)
        h = h + gate_ffn * self.ffn(ffn_in, training=training)

        if update_mask is not None:
            h = update_mask * h + (1.0 - update_mask) * residual

        return h

    @staticmethod
    def modulate(x: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
        """Apply adaLN-Zero affine modulation: ``x * (1 + scale) + shift``."""
        return x * (1.0 + scale) + shift

    @staticmethod
    def ada_ln_bias(width: int, residual_gate_init: float) -> Tensor:
        """Bias vector for adaLN modulation with small nonzero residual gates."""
        zeros = keras.ops.zeros((width,))
        gates = keras.ops.full((width,), residual_gate_init)
        return keras.ops.concatenate([zeros, zeros, gates, zeros, zeros, gates], axis=0)

    @classmethod
    def from_config(cls, config, custom_objects=None):
        return cls(**deserialize(config, custom_objects=custom_objects))

    def get_config(self):
        base_config = layer_kwargs(super().get_config())
        return base_config | serialize(
            {
                "width": self.width,
                "num_heads": self.num_heads,
                "dropout": self.dropout,
                "expansion_factor": self.expansion_factor,
                "glu_variant": self.glu_variant,
                "use_bias": self.use_bias,
                "residual_gate_init": self.residual_gate_init,
                "kernel_initializer": self.kernel_initializer,
            }
        )
