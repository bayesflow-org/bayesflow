import keras

from bayesflow.types import Tensor
from bayesflow.utils import check_lengths_same
from bayesflow.utils.serialization import serializable

from .attention import MultiHeadAttention
from .transformer import Transformer

from ...helpers import Time2Vec, RecurrentEmbedding


@serializable("bayesflow.networks")
class TimeSeriesTransformer(Transformer):
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
        time_embedding: str = "time2vec",
        time_embed_dim: int = 8,
        time_axis: int = None,
        return_sequences: bool = False,
        **kwargs,
    ):
        """(SN) Creates a regular transformer coupled with Time2Vec embeddings of time used to flexibly compress time
        series. If the time intervals vary across batches, it is highly recommended that your simulator also returns a
        "time" vector appended to the simulator outputs and specified via the "time_axis" argument.

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
        time_embedding : str, optional
            The type of embedding to use. Must be in ``["time2vec", "lstm", "gru"]``,
            by default ``"time2vec"``.
        time_embed_dim : int, optional
            Dimensionality of the Time2Vec or recurrent embedding, by default 8.
        time_axis : int or None, optional
            The time axis from which to grab the time vector for the embedding.
            If None, integer time steps 0, 1, ..., sequence_len - 1 are assumed.
        return_sequences : bool, optional
            If True, acts as a many-to-many encoder. If False (default), acts as a
            many-to-one embedding network and ``summary_dim`` is the output dimension.
        **kwargs
            Additional keyword arguments passed to the base layer.
        """

        super().__init__(**kwargs)

        check_lengths_same(embed_dims, num_heads)

        if time_embedding is None:
            self.time_embedding = None
        elif time_embedding == "time2vec":
            self.time_embedding = Time2Vec(num_periodic_features=time_embed_dim - 1)
        elif time_embedding in ["lstm", "gru"]:
            self.time_embedding = RecurrentEmbedding(time_embed_dim, time_embedding)
        else:
            raise ValueError(
                f"Invalid time embedding type: {time_embedding}. Expected one of ['time2vec', 'lstm', 'gru']."
            )

        self.attention_blocks = []
        for i in range(len(embed_dims)):
            block = MultiHeadAttention(
                embed_dim=embed_dims[i],
                num_heads=num_heads[i],
                dropout=dropout,
                expansion_factor=expansion_factor,
                glu_variant=glu_variant,
                kernel_initializer=kernel_initializer,
                use_bias=use_bias,
                layer_norm=layer_norm,
            )
            self.attention_blocks.append(block)

        if return_sequences:
            self.pooling = keras.layers.Identity()
        else:
            self.pooling = keras.layers.GlobalAveragePooling1D()

        self.output_projector = keras.layers.Dense(units=summary_dim)

        self.summary_dim = summary_dim
        self.time_axis = time_axis

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
            Shape ``(batch_size, summary_dim)`` if ``return_sequences=False``, otherwise
            ``(batch_size, sequence_length, summary_dim)``.
        """

        if self.time_axis is not None:
            t = x[..., self.time_axis]
            indices = list(range(keras.ops.shape(x)[-1]))
            indices.pop(self.time_axis)
            inp = keras.ops.take(x, indices, axis=-1)
        else:
            t = None
            inp = x

        if self.time_embedding:
            inp = self.time_embedding(inp, t=t)

        for layer in self.attention_blocks:
            inp = layer(inp, inp, training=training, attention_mask=attention_mask)

        summary = self.pooling(inp)
        summary = self.output_projector(summary)
        return summary
