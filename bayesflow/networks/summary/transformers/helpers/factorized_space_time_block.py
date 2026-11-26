import keras
from keras import ops

from bayesflow.types import Tensor
from bayesflow.utils import layer_kwargs
from bayesflow.utils.serialization import deserialize, serializable, serialize

from ..attention import MultiHeadAttention


@serializable("bayesflow.networks")
class FactorizedSpaceTimeBlock(keras.Layer):
    """Factorized spatial and temporal self-attention for image sequences.

    The spatial block attends over the spatial tokens of every frame
    independently. The temporal block then attends over frames independently
    for every spatial token. This reduces the attention cost from quadratic in
    the full space-time token count to the sum of the spatial and temporal
    attention costs.

    Parameters
    ----------
    embed_dim : int, optional
        Token embedding dimensionality, by default 64.
    num_heads : int, optional
        Number of attention heads in both attention blocks, by default 4.
    dropout : float, optional
        Dropout rate applied to attention outputs and feedforward sublayers, by
        default 0.05.
    expansion_factor : float, optional
        Feedforward intermediate-width multiplier, by default 4.0.
    glu_variant : str, optional
        GLU activation variant for the feedforward sublayers, by default
        ``"swiglu"``.
    kernel_initializer : str, optional
        Initializer for attention and feedforward projection weights, by default
        ``"orthogonal"``.
    use_bias : bool, optional
        Whether projection layers include bias terms, by default False.
    layer_norm : bool, optional
        Whether to apply Pre-LN RMSNorm before each sublayer, by default True.
    gate_attention : bool, optional
        Whether to gate attention residual branches with trainable scalars, by
        default False.
    gate_ffn : bool, optional
        Whether to gate feedforward residual branches with trainable scalars, by
        default True.
    flash_attention : bool or None, optional
        FlashAttention policy forwarded to the underlying attention layers.
    attention_dtype : {"float16", "bfloat16"} or None, optional
        Mixed compute dtype of the underlying attention layers.
    **kwargs
        Additional keyword arguments forwarded to ``keras.Layer``.
    """

    def __init__(
        self,
        embed_dim: int = 64,
        num_heads: int = 4,
        dropout: float = 0.05,
        expansion_factor: float = 4.0,
        glu_variant: str = "swiglu",
        kernel_initializer: str = "orthogonal",
        use_bias: bool = False,
        layer_norm: bool = True,
        gate_attention: bool = False,
        gate_ffn: bool = True,
        flash_attention: bool | None = None,
        attention_dtype: str | None = None,
        **kwargs,
    ):
        super().__init__(**layer_kwargs(kwargs))

        attention_kwargs = {
            "embed_dim": embed_dim,
            "num_heads": num_heads,
            "dropout": dropout,
            "expansion_factor": expansion_factor,
            "glu_variant": glu_variant,
            "kernel_initializer": kernel_initializer,
            "use_bias": use_bias,
            "layer_norm": layer_norm,
            "gate_attention": gate_attention,
            "gate_ffn": gate_ffn,
            "flash_attention": flash_attention,
            "attention_dtype": attention_dtype,
        }
        self.spatial_attention = MultiHeadAttention(**attention_kwargs)
        self.temporal_attention = MultiHeadAttention(**attention_kwargs)

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.expansion_factor = expansion_factor
        self.glu_variant = glu_variant
        self.kernel_initializer = kernel_initializer
        self.use_bias = use_bias
        self.layer_norm = layer_norm
        self.gate_attention = gate_attention
        self.gate_ffn = gate_ffn
        self.flash_attention = flash_attention
        self.attention_dtype = attention_dtype

    def call(
        self,
        x: Tensor,
        training: bool = False,
        attention_mask: Tensor | None = None,
    ) -> Tensor:
        """Apply factorized self-attention to space-time tokens.

        Parameters
        ----------
        x : Tensor
            Input of shape ``(batch_size, num_steps, num_spatial_tokens,
            input_dim)``.
        training : bool, optional
            Passed to dropout and normalization layers, by default False.
        attention_mask : Tensor, optional
            Boolean temporal key-valid mask of shape ``(batch_size,
            num_steps)`` where ``True`` marks an observed frame. The mask is
            shared across spatial tokens.

        Returns
        -------
        Tensor
            Output of shape ``(batch_size, num_steps, num_spatial_tokens,
            embed_dim)``.
        """
        batch_size, num_steps, num_spatial_tokens, input_dim = ops.shape(x)

        temporal_attention_mask = None
        if attention_mask is not None:
            attention_mask = ops.cast(attention_mask, "bool")

            # Share each sequence mask across spatial positions, then fold the
            # spatial axis into the batch axis. The singleton query axis makes
            # this a key-padding mask broadcastable to (B*S, heads, T, T).
            temporal_attention_mask = ops.tile(
                attention_mask[:, None, :],
                (1, num_spatial_tokens, 1),
            )
            temporal_attention_mask = ops.reshape(
                temporal_attention_mask,
                (batch_size * num_spatial_tokens, 1, num_steps),
            )

        # Spatial attention: (B, T, S, D_in) -> (B*T, S, D_in) -> (B*T, S, D).
        x = ops.reshape(
            x,
            (batch_size * num_steps, num_spatial_tokens, input_dim),
        )
        x = self.spatial_attention(x, x, training=training)
        x = ops.reshape(
            x,
            (batch_size, num_steps, num_spatial_tokens, self.embed_dim),
        )

        # Temporal attention: (B, T, S, D) -> (B*S, T, D).
        x = ops.transpose(x, (0, 2, 1, 3))
        x = ops.reshape(
            x,
            (batch_size * num_spatial_tokens, num_steps, self.embed_dim),
        )
        x = self.temporal_attention(
            x,
            x,
            training=training,
            attention_mask=temporal_attention_mask,
        )

        x = ops.reshape(
            x,
            (batch_size, num_spatial_tokens, num_steps, self.embed_dim),
        )
        return ops.transpose(x, (0, 2, 1, 3))

    def build(self, input_shape):
        if self.built:
            return

        spatial_shape = (None, input_shape[2], input_shape[3])
        temporal_shape = (None, input_shape[1], self.embed_dim)
        self.spatial_attention.build(spatial_shape, spatial_shape)
        self.temporal_attention.build(temporal_shape, temporal_shape)
        super().build(input_shape)

    def compute_output_shape(self, input_shape):
        return tuple(input_shape[:-1]) + (self.embed_dim,)

    def get_config(self) -> dict:
        base_config = super().get_config()
        return base_config | serialize(
            {
                "embed_dim": self.embed_dim,
                "num_heads": self.num_heads,
                "dropout": self.dropout,
                "expansion_factor": self.expansion_factor,
                "glu_variant": self.glu_variant,
                "kernel_initializer": self.kernel_initializer,
                "use_bias": self.use_bias,
                "layer_norm": self.layer_norm,
                "gate_attention": self.gate_attention,
                "gate_ffn": self.gate_ffn,
                "flash_attention": self.flash_attention,
                "attention_dtype": self.attention_dtype,
            }
        )

    @classmethod
    def from_config(cls, config, custom_objects=None):
        return cls(**deserialize(config, custom_objects=custom_objects))
