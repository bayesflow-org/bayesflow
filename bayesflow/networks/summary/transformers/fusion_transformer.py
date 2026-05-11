import keras
from keras import layers

from bayesflow.types import Tensor
from bayesflow.utils import check_lengths_same
from bayesflow.utils.serialization import serializable

from .transformer import Transformer
from .attention import MultiHeadAttention


@serializable("bayesflow.networks")
class FusionTransformer(Transformer):
    """
    (SN) Implements a more flexible version of the TimeSeriesTransformer that applies a series of self-attention
    layers followed by cross-attention between the representation and a learnable template summarized via a
    recurrent net.

    Note: This network does not need time embeddings, as the sequence itself is used as a learnable embedding.
    """

    def __init__(
        self,
        summary_dim: int = 16,
        embed_dims: tuple = (64, 64),
        num_heads: tuple = (4, 4),
        dropout: float = 0.05,
        expansion_factor: float = 4.0,
        glu_variant: str = "swiglu",
        kernel_initializer: str = "glorot_uniform",
        use_bias: bool = False,
        layer_norm: bool = True,
        template_type: str = "lstm",
        bidirectional: bool = True,
        template_dim: int = 128,
        **kwargs,
    ):
        """Creates a fusion transformer used to flexibly compress time series.

        Important: This network needs at least 2 transformer blocks and always acts as a many-to-one transform.

        Parameters
        ----------
        summary_dim : int, optional
            Dimensionality of the final summary output, by default 16.
        embed_dims : tuple of int, optional
            Embedding dimensionality for each attention block, by default (64, 64).
        num_heads : tuple of int, optional
            Number of attention heads for each block, by default (4, 4).
        dropout : float, optional
            Dropout rate applied inside the attention sublayer, by default 0.05.
        expansion_factor : float, optional
            FFN intermediate width multiplier (before the 2/3 GLU correction), by default 4.0.
        glu_variant : str, optional
            GLU activation variant for the FFN. One of ``"swiglu"``, ``"geglu"``,
            ``"reglu"``, or ``"liglu"``, by default ``"swiglu"``.
        kernel_initializer : str, optional
            Initializer for kernel weights, by default ``"glorot_uniform"``.
        use_bias : bool, optional
            Whether to include bias terms in dense layers, by default False.
        layer_norm : bool, optional
            Whether to apply Pre-LN RMSNorm before each sublayer, by default True.
        template_type : str, optional
            Recurrent architecture for the template network: ``"lstm"`` or ``"gru"``,
            by default ``"lstm"``.
        bidirectional : bool, optional
            Whether the template recurrent network is bidirectional, by default True.
        template_dim : int, optional
            Hidden units of the recurrent template network, by default 128.
        **kwargs
            Additional keyword arguments passed to the base layer.
        """

        super().__init__(**kwargs)

        check_lengths_same(embed_dims, num_heads)

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
            )
            self.attention_blocks.append(block)

        template_type_upper = template_type.upper()
        if template_type_upper == "LSTM":
            rnn = layers.LSTM(template_dim // 2 if bidirectional else template_dim, dropout=dropout)
        elif template_type_upper == "GRU":
            rnn = layers.GRU(template_dim // 2 if bidirectional else template_dim, dropout=dropout)
        else:
            raise ValueError(f"Argument `template_type` must be 'lstm' or 'gru', got '{template_type}'.")

        self.template_net = layers.Bidirectional(rnn) if bidirectional else rnn

        self.output_projector = keras.layers.Dense(units=summary_dim)
        self.summary_dim = summary_dim

    def call(self, x: Tensor, training: bool = False, attention_mask: Tensor = None) -> Tensor:
        """Compresses the input sequence into a summary vector of size ``summary_dim``.

        Parameters
        ----------
        x : Tensor
            Input of shape ``(batch_size, sequence_length, input_dim)``.
        training : bool, optional
            Passed to dropout and norm layers, by default False.
        attention_mask : Tensor, optional
            Boolean mask of shape ``(B, T, T)`` where 1 = attend, 0 = mask.

        Returns
        -------
        Tensor
            Output of shape ``(batch_size, summary_dim)``.
        """
        template = self.template_net(x, training=training)

        rep = x
        for layer in self.attention_blocks[:-1]:
            rep = layer(rep, rep, training=training, attention_mask=attention_mask)

        summary = self.attention_blocks[-1](keras.ops.expand_dims(template, axis=1), rep, training=training)
        summary = self.output_projector(keras.ops.squeeze(summary, axis=1))
        return summary
