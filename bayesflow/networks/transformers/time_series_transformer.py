import keras

from bayesflow.types import Tensor
from bayesflow.utils import check_lengths_same
from bayesflow.utils.serialization import serializable

from ..embeddings import Time2Vec, RecurrentEmbedding
from ..summary_network import SummaryNetwork

from .mab import MultiHeadAttentionBlock


@serializable("bayesflow.networks")
class TimeSeriesTransformer(SummaryNetwork):
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
        **kwargs : dict
            Additional keyword arguments passed to the base layer.
        """

        super().__init__(**kwargs)

        # Ensure all tuple-settings have the same length
        check_lengths_same(embed_dims, num_heads, mlp_depths, mlp_widths)

        # Initialize Time2Vec embedding layer
        if time_embedding is None:
            self.time_embedding = None
        elif time_embedding == "time2vec":
            self.time_embedding = Time2Vec(num_periodic_features=time_embed_dim - 1)
        elif time_embedding in ["lstm", "gru"]:
            self.time_embedding = RecurrentEmbedding(time_embed_dim, time_embedding)
        else:
            raise ValueError("Embedding not found!")

        # Construct a series of set-attention blocks
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

            block = MultiHeadAttentionBlock(**layer_attention_settings)
            self.attention_blocks.append(block)

        # Pooling will be applied as a final step to the abstract representations obtained from set attention
        self.pooling = keras.layers.GlobalAvgPool1D()
        self.output_projector = keras.layers.Dense(summary_dim)

        self.summary_dim = summary_dim
        self.time_axis = time_axis

    def call(self, input_sequence: Tensor, training: bool = False, **kwargs) -> Tensor:
        """Compresses the input sequence into a summary vector of size `summary_dim`.

        Parameters
        ----------
        input_sequence  : Tensor
            Input of shape (batch_size, sequence_length, input_dim)
        training        : boolean, optional (default - False)
            Passed to the optional internal dropout and spectral normalization
            layers to distinguish between train and test time behavior.
        **kwargs        : dict, optional (default - {})
            Additional keyword arguments passed to the internal attention layer,
            such as ``attention_mask`` or ``return_attention_scores``

        Returns
        -------
        out : Tensor
            Output of shape (batch_size, set_size, output_dim)
        """

        if self.time_axis is not None:
            t = input_sequence[..., self.time_axis]
            indices = list(range(keras.ops.shape(input_sequence)[-1]))
            indices.pop(self.time_axis)
            inp = keras.ops.take(input_sequence, indices, axis=-1)
        else:
            t = None
            inp = input_sequence

        if self.time_embedding:
            inp = self.time_embedding(inp, t=t)

        # Apply self-attention blocks
        for layer in self.attention_blocks:
            inp = layer(inp, inp, training=training, **kwargs)

        # Global average pooling and output projection
        summary = self.pooling(inp)
        summary = self.output_projector(summary)
        return summary
