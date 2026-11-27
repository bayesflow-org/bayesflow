import keras

from bayesflow.networks.helpers import RecurrentEmbedding, Time2Vec
from bayesflow.types import Tensor
from bayesflow.utils import check_lengths_same
from bayesflow.utils.serialization import serializable

from .attention import MultiHeadAttention
from .helpers import FactorizedSpaceTimeBlock, SpatialPositionEmbedding, SummaryToken
from .transformer import Transformer


@serializable("bayesflow.networks")
class SpaceTimeTransformer(Transformer):
    """Summarize ordered sequences of images or other 2D spatial fields.

    A strided 3D convolution embeds overlapping spatial patches and captures
    local motion across neighboring frames. Its temporal stride is one, so each
    input step has a corresponding set of patches. Factorized attention mixes
    patches within each frame and across time. A learned query reads out the
    features per frame or across the full sequence.

    Parameters
    ----------
    summary_dim : int, optional
        Output width in both summary and sequence mode, by default 16.
    embed_dims : tuple of int, optional
        Embedding width for each factorized attention block, by default ``(64, 64)``.
    num_heads : tuple of int, optional
        Number of attention heads for each block, by default ``(4, 4)``.
    dropout : float, optional
        Dropout rate in the attention blocks, by default 0.05.
    expansion_factor : float, optional
        Feedforward width multiplier, by default 4.0.
    glu_variant : str, optional
        GLU activation variant for the feedforward layers, by default ``"swiglu"``.
    kernel_initializer : str, optional
        Initializer for kernel weights, by default ``"orthogonal"``.
    use_bias : bool, optional
        Whether to include bias terms in projections, by default False.
    layer_norm : bool, optional
        Whether to apply RMSNorm, by default True.
    gate_attention : bool, optional
        Whether to gate attention residual branches, by default False.
    gate_ffn : bool, optional
        Whether to gate feedforward residual branches, by default True.
    time_embedding : str or None, optional
        Embedding for frame positions: ``"time2vec"`` (default), ``"lstm"``, or
        ``"gru"``. If None, raw frame positions are concatenated to the features.
    time_embed_dim : int, optional
        Width of the time embedding, by default 8.
    patch_size : int, optional
        Spatial stride of the patch embedding, by default 8. The spatial kernel
        is one pixel wider than the stride, and images are padded as needed.
    return_sequences : bool, optional
        If True, apply the learned readout independently to the patches of each
        frame. By default, apply it once to the patches of all valid frames.
    flash_attention : bool or None, optional
        FlashAttention policy passed to :class:`MultiHeadAttention`.
    attention_dtype : {"float16", "bfloat16"} or None, optional
        Mixed compute dtype of the underlying Keras attention layers.
    temporal_kernel_size : int, optional
        Number of neighboring frames covered by the patch embedding, by default 3.
        Use 1 to embed frames independently before attention.
    **kwargs
        Additional arguments forwarded to :class:`SummaryNetwork`. The optional
        ``base_distribution`` is supported in summary mode.

    References
    ----------
    Bertasius et al. (2021). Is Space-Time Attention All You Need for Video Understanding?
    https://proceedings.mlr.press/v139/bertasius21a.html

    Yan et al. (2025). Learning Streaming Video Representation via Multitask Training.
    https://arxiv.org/abs/2504.20041
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
        gate_attention: bool = False,
        gate_ffn: bool = True,
        time_embedding: str | None = "time2vec",
        time_embed_dim: int = 8,
        patch_size: int = 8,
        return_sequences: bool = False,
        flash_attention: bool | None = None,
        attention_dtype: str | None = None,
        temporal_kernel_size: int = 3,
        **kwargs,
    ):
        super().__init__(**kwargs)

        check_lengths_same(embed_dims, num_heads)

        if time_embedding is None:
            self.time_embedding = None
        elif time_embedding == "time2vec":
            self.time_embedding = Time2Vec(num_periodic_features=time_embed_dim - 1)
        elif time_embedding in ("lstm", "gru"):
            self.time_embedding = RecurrentEmbedding(time_embed_dim, time_embedding)
        else:
            raise ValueError(f"Unknown time embedding {time_embedding!r}. Use 'time2vec', 'lstm', 'gru', or None.")

        self.summary_dim = summary_dim
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.dropout_rate = dropout
        self.expansion_factor = expansion_factor
        self.glu_variant = glu_variant
        self.kernel_initializer = kernel_initializer
        self.use_bias = use_bias
        self.layer_norm = layer_norm
        self.gate_attention = gate_attention
        self.gate_ffn = gate_ffn
        self.time_embedding_type = time_embedding
        self.time_embed_dim = time_embed_dim
        self.patch_size = patch_size
        self.return_sequences = return_sequences
        self.flash_attention = flash_attention
        self.attention_dtype = attention_dtype
        self.temporal_kernel_size = temporal_kernel_size

        spatial_kernel_size = patch_size + 1
        self.patch_embedding = keras.layers.Conv3D(
            filters=embed_dims[0],
            kernel_size=(temporal_kernel_size, spatial_kernel_size, spatial_kernel_size),
            strides=(1, patch_size, patch_size),
            padding="same",
            kernel_initializer=kernel_initializer,
            use_bias=use_bias,
        )
        self.spatial_positions = SpatialPositionEmbedding()

        attention_kwargs = dict(
            dropout=dropout,
            expansion_factor=expansion_factor,
            glu_variant=glu_variant,
            kernel_initializer=kernel_initializer,
            use_bias=use_bias,
            layer_norm=layer_norm,
            gate_attention=gate_attention,
            gate_ffn=gate_ffn,
            flash_attention=flash_attention,
            attention_dtype=attention_dtype,
        )
        self.attention_blocks = [
            FactorizedSpaceTimeBlock(embed_dim=width, num_heads=heads, **attention_kwargs)
            for width, heads in zip(embed_dims, num_heads)
        ]

        self.summary_token = SummaryToken(kernel_initializer=kernel_initializer)
        self.summary_attention = MultiHeadAttention(
            embed_dim=embed_dims[-1], num_heads=num_heads[-1], **attention_kwargs
        )

        self.output_norm = keras.layers.RMSNormalization() if layer_norm else None
        self.output_projector = keras.layers.Dense(
            summary_dim, kernel_initializer=kernel_initializer, use_bias=use_bias
        )

    def call(
        self, x: Tensor, training: bool = False, attention_mask: Tensor | None = None, mask: Tensor | None = None
    ) -> Tensor:
        """Encode an ordered sequence of images or other 2D spatial fields.

        Parameters
        ----------
        x : Tensor
            Input of shape ``(batch_size, time_steps, height, width, channels)``.
        training : bool, optional
            Passed to dropout and normalization layers, by default False.
        attention_mask : Tensor, optional
            Boolean mask of shape ``(batch_size, time_steps)``. ``True`` marks
            a valid frame. Takes precedence over ``mask``.
        mask : Tensor, optional
            Frame mask used when ``attention_mask`` is None. Has the same shape
            and meaning as ``attention_mask``.

        Returns
        -------
        Tensor
            A summary of shape ``(batch_size, summary_dim)`` when
            ``return_sequences=False``. With ``return_sequences=True``, returns
            ``(batch_size, time_steps, summary_dim)`` with zeros at masked steps.
        """
        frame_mask = mask if attention_mask is None else attention_mask
        if frame_mask is not None:
            frame_mask = keras.ops.cast(frame_mask, "bool")
            x = keras.ops.where(frame_mask[..., None, None, None], x, 0)

        tokens = self._embed_patches(x)
        tokens = self._embed_time(tokens, training=training)

        for block in self.attention_blocks:
            tokens = block(tokens, training=training, attention_mask=frame_mask)

        summary = self._readout(tokens, frame_mask, training=training)

        if self.output_norm is not None:
            summary = self.output_norm(summary, training=training)

        outputs = self.output_projector(summary)
        if self.return_sequences and frame_mask is not None:
            outputs = keras.ops.where(frame_mask[..., None], outputs, 0)

        return outputs

    def build(self, input_shape):
        # Only the channel count affects the weights. Use a minimal input to
        # let SummaryNetwork build the layers even if steps and resolution are unknown.
        _, _, _, _, channels = input_shape
        super().build((1, 1, 1, 1, channels))

    def compute_output_shape(self, input_shape):
        batch, steps, _, _, _ = input_shape
        return (batch, steps, self.summary_dim) if self.return_sequences else (batch, self.summary_dim)

    def compute_mask(self, inputs, mask=None):
        return None

    def _embed_patches(self, x: Tensor) -> Tensor:
        patches = self.spatial_positions(self.patch_embedding(x))
        batch, steps, rows, columns, embed_dim = keras.ops.shape(patches)

        return keras.ops.reshape(patches, (batch, steps, rows * columns, embed_dim))

    def _embed_time(self, tokens: Tensor, training: bool) -> Tensor:
        batch, steps, patches, _ = keras.ops.shape(tokens)

        if self.time_embedding is None:
            positions = keras.ops.arange(steps, dtype=tokens.dtype)[None, :, None]
        else:
            # The embedding layers append positions to their input features.
            empty_features = keras.ops.zeros((1, steps, 0), dtype=tokens.dtype)
            positions = self.time_embedding(empty_features, training=training)

        time_width = keras.ops.shape(positions)[-1]
        positions = keras.ops.broadcast_to(positions[:, :, None, :], (batch, steps, patches, time_width))
        return keras.ops.concatenate([tokens, positions], axis=-1)

    def _readout(self, tokens: Tensor, frame_mask: Tensor | None, training: bool) -> Tensor:
        batch, steps, patches, embed_dim = keras.ops.shape(tokens)
        key_mask = None

        if frame_mask is not None:
            tokens = keras.ops.where(frame_mask[..., None, None], tokens, 0)

        if self.return_sequences:
            context = keras.ops.reshape(tokens, (batch * steps, patches, embed_dim))
        else:
            context = keras.ops.reshape(tokens, (batch, steps * patches, embed_dim))

            if frame_mask is not None:
                key_mask = keras.ops.repeat(frame_mask, patches, axis=1)[:, None, :]

        context = self.summary_token(context)
        key_mask = self.summary_token.update_mask(key_mask)
        readout = self.summary_attention(context[:, -1:, :], context, training=training, attention_mask=key_mask)
        readout = self.summary_token.take(readout)

        if self.return_sequences:
            readout = keras.ops.reshape(readout, (batch, steps, embed_dim))

        return readout
