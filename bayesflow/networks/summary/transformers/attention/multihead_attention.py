import keras
from keras import layers

from bayesflow.types import Tensor
from bayesflow.utils import layer_kwargs
from bayesflow.utils.serialization import serializable, serialize, deserialize

from bayesflow.networks.helpers import FFN


@serializable("bayesflow.networks")
class MultiHeadAttention(keras.Layer):
    """Pre-LN multi-head attention (MAB) block with a GLU feedforward sublayer.

    Implements the MAB block from [1] with SOTA improvements:
    - Pre-LN via RMSNorm applied before each sublayer for stable training [2]
    - Clean additive residual paths with no post-normalization
    - Bias-free projections by default
    - GLU-family feedforward network (SwiGLU default) [3]

    [1] Lee, J., Lee, Y., Kim, J., Kosiorek, A., Choi, S., & Teh, Y. W. (2019).
        Set transformer: A framework for attention-based permutation-invariant neural networks.
        In International conference on machine learning (pp. 3744-3753). PMLR.
    [2] Xiong, R. et al. (2020). On layer normalization in the transformer architecture. ICML.
    [3] Shazeer, N. (2020). GLU variants improve transformer. arXiv:2002.05202.
    """

    def __init__(
        self,
        embed_dim: int = 64,
        num_heads: int = 4,
        dropout: float = 0.05,
        expansion_factor: float = 4.0,
        glu_variant: str = "swiglu",
        kernel_initializer: str = "glorot_uniform",
        use_bias: bool = False,
        layer_norm: bool = True,
        **kwargs,
    ):
        """
        Parameters
        ----------
        embed_dim : int, optional
            Dimensionality of the embedding space, by default 64.
        num_heads : int, optional
            Number of attention heads, by default 4.
        dropout : float, optional
            Dropout rate applied inside the attention sublayer, by default 0.05.
        expansion_factor : float, optional
            FFN intermediate width multiplier (before the 2/3 GLU correction),
            by default 4.0.
        glu_variant : str, optional
            GLU activation variant for the FFN. One of ``"swiglu"``, ``"geglu"``,
            ``"reglu"``, or ``"liglu"``, by default ``"swiglu"``.
        kernel_initializer : str, optional
            Weight initializer for all Dense projections, by default
            ``"glorot_uniform"``.
        use_bias : bool, optional
            Whether to include bias terms in all Dense projections, by default
            False (SOTA practice for transformer blocks).
        layer_norm : bool, optional
            Whether to apply Pre-LN RMSNorm before each sublayer, by default True.
        **kwargs
            Additional keyword arguments passed to ``keras.Layer``.
        """
        super().__init__(**layer_kwargs(kwargs))

        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim={embed_dim} must be divisible by num_heads={num_heads}.")

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout_rate = dropout
        self.expansion_factor = expansion_factor
        self.glu_variant = glu_variant
        self.kernel_initializer = kernel_initializer
        self.use_bias = use_bias
        self.layer_norm = layer_norm

        self.input_projector = layers.Dense(units=embed_dim, use_bias=use_bias, kernel_initializer=kernel_initializer)

        self.attention = layers.MultiHeadAttention(
            key_dim=embed_dim // num_heads,
            num_heads=num_heads,
            dropout=dropout,
            use_bias=use_bias,
            output_shape=embed_dim,
            kernel_initializer=kernel_initializer,
        )

        self.feedforward = FFN(
            embed_dim=embed_dim,
            expansion_factor=expansion_factor,
            glu_variant=glu_variant,
            use_bias=use_bias,
            dropout=dropout,
            kernel_initializer=kernel_initializer,
        )

        self.ln_attn = layers.RMSNormalization() if layer_norm else None
        self.ln_kv = layers.RMSNormalization() if layer_norm else None
        self.ln_ffn = layers.RMSNormalization() if layer_norm else None

    def call(
        self,
        x: Tensor,
        y: Tensor,
        training: bool = False,
        attention_mask: Tensor = None,
    ) -> Tensor:
        """Performs the forward pass through the Pre-LN attention block.

        Parameters
        ----------
        x : Tensor
            Query input of shape ``(batch_size, seq_len_x, input_dim)``.
        y : Tensor
            Key/value input of shape ``(batch_size, seq_len_y, input_dim)``.
        training : bool, optional
            Toggles dropout and norm training behaviour, by default False.
        attention_mask : Tensor, optional
            Boolean mask of shape ``(B, T, T)`` where 1 = attend, 0 = mask.

        Returns
        -------
        Tensor
            Output of shape ``(batch_size, seq_len_x, embed_dim)``.
        """
        x = self.input_projector(x)

        # Attention sublayer: Pre-LN -> attention -> residual
        query = self.ln_attn(x, training=training) if self.ln_attn is not None else x
        key_value = self.ln_kv(y, training=training) if self.ln_kv is not None else y
        x = x + self.attention(
            query=query, key=key_value, value=key_value, training=training, attention_mask=attention_mask
        )

        # FFN sublayer: Pre-LN -> FFN -> residual
        ffn_in = self.ln_ffn(x, training=training) if self.ln_ffn is not None else x
        x = x + self.feedforward(ffn_in, training=training)

        return x

    def build(self, x_shape, y_shape):
        if self.built:
            return

        self.input_projector.build(x_shape)
        proj_shape = self.input_projector.compute_output_shape(x_shape)

        if self.ln_attn is not None:
            self.ln_attn.build(proj_shape)
        if self.ln_kv is not None:
            self.ln_kv.build(y_shape)

        self.attention.build(proj_shape, y_shape, y_shape)

        attn_out_shape = proj_shape

        if self.ln_ffn is not None:
            self.ln_ffn.build(attn_out_shape)

        self.feedforward.build(attn_out_shape)

    def compute_output_shape(self, x_shape, y_shape):
        return tuple(x_shape)[:-1] + (self.embed_dim,)

    def get_config(self) -> dict:
        base_config = super().get_config()
        return base_config | serialize(
            {
                "embed_dim": self.embed_dim,
                "num_heads": self.num_heads,
                "dropout": self.dropout_rate,
                "expansion_factor": self.expansion_factor,
                "glu_variant": self.glu_variant,
                "kernel_initializer": self.kernel_initializer,
                "use_bias": self.use_bias,
                "layer_norm": self.layer_norm,
            }
        )

    @classmethod
    def from_config(cls, config, custom_objects=None):
        return cls(**deserialize(config, custom_objects=custom_objects))
