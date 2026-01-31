import keras

from bayesflow.types import Tensor
from bayesflow.utils import check_lengths_same
from bayesflow.utils.serialization import serializable

from .attention import MultiHeadAttention
from .transformer import Transformer

from ..embeddings import Time2Vec, RecurrentEmbedding


@serializable("bayesflow.networks")
class TimeSeriesTransformer(Transformer):
    def __init__(
        self,
        summary_dim: int = 16,
        embed_dims: tuple = (64, 64),
        num_heads: tuple = (4, 4),
        mlp_depths: tuple = (2, 2),
        mlp_widths: tuple = (128, 128),
        dropout: float = 0.05,
        mlp_activation: str = "gelu",
        kernel_initializer: str = "lecun_normal",
        use_bias: bool = True,
        layer_norm: bool = True,
        time_embedding: str = "time2vec",
        time_embed_dim: int = 8,
        time_axis: int = None,
        many_to_one: bool = True,
        **kwargs,
    ):
        """(SN) Creates a regular transformer coupled with Time2Vec embeddings of time used to flexibly compress time
        series. If the time intervals vary across batches, it is highly recommended that your simulator also returns a
        "time" vector appended to the simulator outputs and specified via the "time_axis" argument.

        Parameters
        ----------
        summary_dim : int, optional (default - 16)
            Dimensionality of the final summary output.
        embed_dims  : tuple of int, optional (default - (64, 64))
            Dimensions of the keys, values, and queries for each attention block.
        num_heads   : tuple of int, optional (default - (4, 4))
            Number of attention heads for each embedding dimension.
        mlp_depths  : tuple of int, optional (default - (2, 2))
            Depth of the multi-layer perceptron (MLP) blocks for each component.
        mlp_widths  : tuple of int, optional (default - (128, 128))
            Width of each MLP layer in each block for each component.
        dropout     : float, optional (default - 0.05)
            Dropout rate applied to the attention and MLP layers. If set to None, no dropout is applied.
        mlp_activation : str, optional (default - 'gelu')
            Activation function used in the dense layers. Common choices include "relu", "elu", and "gelu".
        kernel_initializer : str, optional (default - 'lecun_normal')
            Initializer for the kernel weights matrix. Common choices include "glorot_uniform", "he_normal", etc.
        use_bias : bool, optional (default - True)
            Whether to include a bias term in the dense layers.
        layer_norm : bool, optional (default - True)
            Whether to apply layer normalization after the attention and MLP layers.
        time_embedding  : str, optional (default - "time2vec")
            The type of embedding to use. Must be in ["time2vec", "lstm", "gru"]
        time_embed_dim  : int, optional (default - 8)
            The dimensionality of the Time2Vec or recurrent embedding.
        time_axis     : int, optional (default - None)
            The time axis (e.g., -1 for last axis) from which to grab the time vector that goes into the embedding.
            If an embedding is provided and time_axis is None, a uniform time interval between [0, sequence_len]
            will be assumed.
        many_to_one   : bool, optional (default - True)
            If True, acts as a many-to-one embedding network (to be used for compression tasks).
            If False, acts as a many-to-many encoder (to be used for time-varying tasks).
            In that case, the `summary_dim` argument denotes the dimension of the output sequence.
        **kwargs : dict
            Additional keyword arguments passed to the base layer.
        """

        super().__init__(**kwargs)

        check_lengths_same(embed_dims, num_heads, mlp_depths, mlp_widths)

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
            layer_attention_settings = dict(
                dropout=dropout,
                mlp_activation=mlp_activation,
                kernel_initializer=kernel_initializer,
                use_bias=use_bias,
                layer_norm=layer_norm,
                num_heads=num_heads[i],
                embed_dim=embed_dims[i],
                mlp_depth=mlp_depths[i],
                mlp_width=mlp_widths[i],
            )

            block = MultiHeadAttention(**layer_attention_settings)
            self.attention_blocks.append(block)

        if many_to_one:
            self.pooling = keras.layers.GlobalAvgPool1D()
        else:
            self.pooling = keras.layers.Identity()

        self.output_projector = keras.layers.Dense(units=summary_dim)

        self.summary_dim = summary_dim
        self.time_axis = time_axis

    def call(self, x: Tensor, training: bool = False, attention_mask: Tensor = None) -> Tensor:
        """Compresses the input sequence into a summary vector of size `summary_dim`.

        Parameters
        ----------
        x               : Tensor
            Input of shape (batch_size, sequence_length, input_dim)
        training        : boolean, optional (default - False)
            Passed to the optional internal dropout and spectral normalization
            layers to distinguish between train and test time behavior.
        attention_mask  : a boolean mask of shape `(B, T, T)`, that prevents
            attention to certain positions. The boolean mask specifies which
            query elements can attend to which key elements, 1 indicates
            attention and 0 indicates no attention. Broadcasting can happen for
            the missing batch dimensions and the head dimension.

        Returns
        -------
        out : Tensor
            Output of shape (batch_size, summary_dim) if `many_to_one=True`, otherwise
            a new sequence of shape (batch_size, sequence_length, summary_dim).
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

        # Apply self-attention blocks
        for layer in self.attention_blocks:
            inp = layer(inp, inp, training=training, attention_mask=attention_mask)

        # Global average pooling and output projection
        summary = self.pooling(inp)
        summary = self.output_projector(summary)
        return summary
