import keras

from bayesflow.types import Tensor
from bayesflow.utils import check_lengths_same, expand_tile
from bayesflow.utils.serialization import serializable, serialize

from .attention import MultiHeadAttention
from .helpers import Downsample, SummaryToken
from .transformer import Transformer

from ...helpers import Time2Vec, RecurrentEmbedding


@serializable("bayesflow.networks")
class TimeSeriesTransformer(Transformer):
    """Transformer summary network for time series.

    Couples self-attention blocks with optional time embeddings to compress time
    series. If time intervals vary across batches, the simulator should return a
    time vector appended to the simulator outputs and specify it via ``time_axis``.

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
    time_embedding : str, optional
        Time embedding type. Must be one of ``"time2vec"``, ``"lstm"``, ``"gru"``,
        or None. If None, raw time values are concatenated to the sequence features.
    time_embed_dim : int, optional
        Dimensionality of the time embedding, by default 8.
    time_axis : int or None, optional
        Feature axis containing explicit time values. If None, integer positions
        are used.
    downsample : int or None, optional
        Optional temporal downsampling factor applied before the transformer blocks.
        If None, an identity layer is used. If an integer greater than one, a strided
        ``Conv1D`` reduces the sequence length by the requested factor.
    return_sequences : bool, optional
        Whether to return one summary per time step. If False, returns a single
        summary vector from a learned summary token appended to the sequence.
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
        time_embedding: str = "time2vec",
        time_embed_dim: int = 8,
        time_axis: int | None = None,
        downsample: int | None = None,
        return_sequences: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        check_lengths_same(embed_dims, num_heads)
        if isinstance(downsample, bool) or (
            downsample is not None and (not isinstance(downsample, int) or downsample < 1)
        ):
            raise ValueError(f"downsample must be None or a positive integer, got {downsample!r}.")
        downsample = None if downsample in (None, 1) else downsample

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
                dropout=dropout,
                expansion_factor=expansion_factor,
                glu_variant=glu_variant,
                kernel_initializer=kernel_initializer,
                use_bias=use_bias,
                layer_norm=layer_norm,
                gate_attention=gate_attention,
                gate_ffn=gate_ffn,
            )
            self.attention_blocks.append(block)

        self.summary_token = None if return_sequences else SummaryToken(kernel_initializer=kernel_initializer)

        self.output_projector = keras.layers.Dense(
            units=summary_dim,
            kernel_initializer=kernel_initializer,
        )

        self.summary_dim = summary_dim
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.dropout_rate = dropout
        self.expansion_factor = expansion_factor
        self.glu_variant = glu_variant
        self.kernel_initializer = kernel_initializer
        self.use_bias = use_bias
        self.layer_norm = layer_norm
        self.time_embedding_type = time_embedding
        self.time_embed_dim = time_embed_dim
        self.time_axis = time_axis
        self.downsample = downsample
        self.return_sequences = return_sequences
        self.gate_attention = gate_attention
        self.gate_ffn = gate_ffn

    @staticmethod
    def make_default_time(x: Tensor) -> Tensor:
        t = keras.ops.arange(keras.ops.shape(x)[1], dtype=x.dtype)
        return expand_tile(t, keras.ops.shape(x)[0], axis=0)

    @staticmethod
    def make_attention_mask(attention_mask: Tensor | None = None, mask: Tensor | None = None) -> Tensor | None:
        if attention_mask is None:
            attention_mask = mask

        if attention_mask is None:
            return None

        if len(attention_mask.shape) != 2:
            raise ValueError(f"Expected mask with shape (batch, sequence_length), got {attention_mask.shape}.")

        attention_mask = keras.ops.cast(attention_mask, "bool")
        # key-padding mask; keras broadcasts (B, 1, T) over heads and query steps
        return keras.ops.expand_dims(attention_mask, axis=1)

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
            Boolean sequence mask of shape ``(B, T)`` where 1 = observed feature
            token and 0 = padded or missing feature token. Takes precedence over
            any mask derived from ``mask``.
        mask : Tensor, optional
            Boolean sequence mask of shape ``(B, T)`` where 1 = observed feature
            token and 0 = padded or missing feature token. If ``attention_mask`` is
            not provided, this is converted to a key-padding attention mask before
            downsampling. Explicit time values are downsampled separately and are
            not masked by this feature-token mask.

        Returns
        -------
        Tensor
            Shape ``(batch_size, summary_dim)`` if ``return_sequences=False``, otherwise
            ``(batch_size, sequence_length, summary_dim)`` or the corresponding
            downsampled sequence length when ``downsample`` is set.
        """

        if self.time_axis is not None:
            time_vec = x[..., self.time_axis]
            indices = list(range(keras.ops.shape(x)[-1]))
            indices.pop(self.time_axis)
            inp = keras.ops.take(x, indices, axis=-1)
        else:
            time_vec = self.make_default_time(x)
            inp = x

        attention_mask = self.make_attention_mask(attention_mask=attention_mask, mask=mask)

        inp = self.downsampler(inp)
        attention_mask = self.downsampler.downsample_mask(attention_mask)
        time_vec = self.downsampler.downsample_time(time_vec)

        if self.time_embedding is not None:
            inp = self.time_embedding(inp, t=time_vec)
        else:
            inp = keras.ops.concatenate([inp, time_vec[..., None]], axis=-1)

        if not self.return_sequences:
            inp = self.summary_token(inp)
            attention_mask = self.summary_token.update_mask(attention_mask)

        for layer in self.attention_blocks:
            inp = layer(inp, inp, training=training, attention_mask=attention_mask)

        if self.return_sequences:
            # sequence returned unreduced so caller needs to mask the padded steps
            summary = inp
        else:
            summary = self.summary_token.take(inp)
        summary = self.output_projector(summary)
        return summary

    def compute_mask(self, inputs, mask=None):
        # `mask` (magic keyword in Keras) is terminated here by `return None`
        # to prevent warnings about inability to inject it downstream.
        # We explicitly pass mask and do not rely on having it travel with as a tensor attribute.
        return None

    def get_config(self) -> dict:
        base_config = super().get_config()
        return base_config | serialize(
            {
                "summary_dim": self.summary_dim,
                "embed_dims": self.embed_dims,
                "num_heads": self.num_heads,
                "dropout": self.dropout_rate,
                "expansion_factor": self.expansion_factor,
                "glu_variant": self.glu_variant,
                "kernel_initializer": self.kernel_initializer,
                "use_bias": self.use_bias,
                "layer_norm": self.layer_norm,
                "gate_attention": self.gate_attention,
                "gate_ffn": self.gate_ffn,
                "time_embedding": self.time_embedding_type,
                "time_embed_dim": self.time_embed_dim,
                "time_axis": self.time_axis,
                "downsample": self.downsample,
                "return_sequences": self.return_sequences,
            }
        )
