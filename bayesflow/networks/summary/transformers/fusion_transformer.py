import keras
from keras import layers

from bayesflow.types import Tensor
from bayesflow.utils import check_lengths_same
from bayesflow.utils.serialization import serializable

from .transformer import Transformer
from .attention import MultiHeadAttention
from .helpers import Downsample


@serializable("bayesflow.networks")
class FusionTransformer(Transformer):
    """Fusion transformer summary network for time series.

    Applies a series of self-attention layers followed by cross-attention between
    the representation and a learnable recurrent template. This network does not
    use explicit time embeddings because the sequence itself is used as a
    learnable embedding.

    Important: This network needs at least two transformer blocks and always acts
    as a many-to-one transform.

    Parameters
    ----------
    summary_dim : int, optional
        Dimensionality of the final summary output, by default 16.
    embed_dims : tuple of int, optional
        Embedding dimensionality for each attention block, by default ``(64, 64)``.
    num_heads : tuple of int, optional
        Number of attention heads for each block, by default ``(4, 4)``.
    dropout : float, optional
        Dropout rate applied inside attention sublayers, by default 0.05.
    expansion_factor : float, optional
        FFN intermediate width multiplier, by default 4.0.
    glu_variant : str, optional
        GLU activation variant for the FFN, by default ``"swiglu"``.
    kernel_initializer : str, optional
        Initializer for kernel weights, by default ``"glorot_uniform"``.
    use_bias : bool, optional
        Whether to include bias terms in dense layers, by default False.
    layer_norm : bool, optional
        Whether to apply Pre-LN RMSNorm before each sublayer, by default True.
    gate_attention : bool, optional
        Whether to gate attention residual branches, by default True.
    gate_ffn : bool, optional
        Whether to gate feedforward residual branches, by default True.
    template_type : str, optional
        Recurrent architecture for the template network, by default ``"lstm"``.
    bidirectional : bool, optional
        Whether the template recurrent network is bidirectional, by default True.
    template_dim : int, optional
        Hidden units of the recurrent template network, by default 128.
    downsample : int or None, optional
        Optional temporal downsampling factor applied before the recurrent
        template and transformer blocks. If None, no downsampling is applied.
    """

    def __init__(
        self,
        summary_dim: int = 16,
        embed_dims: tuple = (64, 64),
        num_heads: tuple = (4, 4),
        dropout: float = 0.05,
        expansion_factor: float = 4.0,
        glu_variant: str = "swiglu",
        kernel_initializer: str = "orthogonal",
        use_bias: bool = False,
        layer_norm: bool = True,
        gate_attention: bool = True,
        gate_ffn: bool = True,
        template_type: str = "lstm",
        bidirectional: bool = True,
        template_dim: int = 128,
        downsample: int | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        check_lengths_same(embed_dims, num_heads)
        if isinstance(downsample, bool) or (
            downsample is not None and (not isinstance(downsample, int) or downsample < 1)
        ):
            raise ValueError(f"downsample must be None or a positive integer, got {downsample!r}.")
        downsample = None if downsample in (None, 1) else downsample

        self.downsampler = Downsample(
            factor=downsample,
            filters=embed_dims[0],
            kernel_initializer=kernel_initializer,
            use_bias=use_bias,
        )

        self.attention_blocks = []
        for i in range(len(embed_dims)):
            block = MultiHeadAttention(
                embed_dim=embed_dims[i],
                num_heads=num_heads[i],
                expansion_factor=expansion_factor,
                glu_variant=glu_variant,
                dropout=dropout,
                kernel_initializer=kernel_initializer,
                use_bias=use_bias,
                layer_norm=layer_norm,
                gate_attention=gate_attention,
                gate_ffn=gate_ffn,
            )
            self.attention_blocks.append(block)

        template_type_upper = template_type.upper()
        if template_type_upper == "LSTM":
            rnn = layers.LSTM(template_dim)
        elif template_type_upper == "GRU":
            rnn = layers.GRU(template_dim)
        else:
            raise ValueError(f"Argument `template_type` must be 'lstm' or 'gru', got '{template_type}'.")

        self.template_net = layers.Bidirectional(rnn, merge_mode="sum") if bidirectional else rnn

        self.output_projector = keras.layers.Dense(
            units=summary_dim,
            kernel_initializer=kernel_initializer,
        )
        self.dropout_layer = keras.layers.Dropout(dropout)
        self.summary_dim = summary_dim
        self.gate_attention = gate_attention
        self.gate_ffn = gate_ffn
        self.downsample = downsample

    def call(
        self, x: Tensor, training: bool = False, attention_mask: Tensor | None = None, mask: Tensor | None = None
    ) -> Tensor:
        """Compresses the input sequence into a summary vector of size ``summary_dim``.

        Parameters
        ----------
        x : Tensor
            Input of shape ``(batch_size, sequence_length, input_dim)``.
        training : bool, optional
            Passed to dropout and norm layers, by default False.
        attention_mask : Tensor, optional
            Boolean mask broadcastable to ``(B, num_heads, T, T)`` where 1 = attend,
            0 = mask. Takes precedence over any mask derived from ``mask``.
        mask : Tensor, optional
            Boolean padding mask of shape ``(B, T)`` where 1 = real time step,
            0 = padding. Used for variable-length trajectories padded to a common
            length: it masks the recurrent template, builds a key-padding
            ``attention_mask`` (when none is given), and excludes padded steps
            from the final cross-attention.

        Returns
        -------
        Tensor
            Output of shape ``(batch_size, summary_dim)``.
        """
        if attention_mask is None and mask is not None:
            # key-padding mask; keras broadcasts (B, 1, T) over heads and query steps
            attention_mask = keras.ops.expand_dims(keras.ops.cast(mask, "bool"), axis=1)

        if self.downsample is not None and attention_mask is not None:
            if len(attention_mask.shape) == 2:
                attention_mask = keras.ops.expand_dims(attention_mask, axis=1)
            if len(attention_mask.shape) != 3 or attention_mask.shape[-2] != 1:
                raise ValueError(
                    "Downsampled FusionTransformer only supports key-padding attention masks "
                    f"with shape (B, T) or (B, 1, T), got {attention_mask.shape}."
                )
            attention_mask = self.downsampler.downsample_mask(attention_mask)

        if self.downsample is not None and mask is not None:
            recurrent_mask = keras.ops.expand_dims(keras.ops.cast(mask, "bool"), axis=1)
            recurrent_mask = self.downsampler.downsample_mask(recurrent_mask)
            mask = keras.ops.squeeze(recurrent_mask, axis=1)

        x = self.downsampler(x)
        template = self.template_net(x, training=training, mask=mask)
        template = self.dropout_layer(template, training=training)

        rep = x
        for layer in self.attention_blocks[:-1]:
            rep = layer(rep, rep, training=training, attention_mask=attention_mask)

        summary = self.attention_blocks[-1](
            keras.ops.expand_dims(template, axis=1),
            rep,
            training=training,
            attention_mask=attention_mask,
        )
        summary = self.output_projector(keras.ops.squeeze(summary, axis=1))
        return summary

    def compute_mask(self, inputs, mask=None):
        # `mask` (magic keyword in Keras) is terminated here by `return None`
        # to prevent warnings about inability to inject it downstream.
        # We explicitly pass mask and do not rely on having it travel with as a tensor attribute.
        return None
