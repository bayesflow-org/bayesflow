import keras
from keras.saving import register_keras_serializable as serializable

from bayesflow.types import Tensor
from bayesflow.utils import check_lengths_same

from ..embeddings import Time2Vec
from ..summary_network import SummaryNetwork

from .mab import MultiHeadAttentionBlock


@serializable(package="bayesflow.networks")
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
        kernel_initializer: str = "he_normal",
        use_bias: bool = True,
        layer_norm: bool = True,
        t2v_embed_dim: int = 8,
        **kwargs,
    ):
        """Creates a regular transformer coupled with Time2Vec embeddings of time used to flexibly compress time series.
        If the time intervals vary across batches, it is highly recommended that your simulator also returns a "time"
        vector denoting absolute or relative time.

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
        kernel_initializer : str, optional (default - 'he_normal')
            Initializer for the kernel weights matrix. Common choices include "glorot_uniform", "he_normal", etc.
        use_bias : bool, optional (default - True)
            Whether to include a bias term in the dense layers.
        layer_norm : bool, optional (default - True)
            Whether to apply layer normalization after the attention and MLP layers.
        t2v_embed_dim : int, optional (default - 8)
            The dimensionality of the Time2Vec embedding.
        **kwargs : dict
            Additional keyword arguments passed to the base layer.
        """

        super().__init__(**kwargs)

        # Ensure all tuple-settings have the same length
        check_lengths_same(embed_dims, num_heads, mlp_depths, mlp_widths)

        # Initialize Time2Vec embedding layer
        self.time2vec = Time2Vec(t2v_embed_dim)

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

    def call(self, input_sequence: Tensor, time: Tensor = None, training: bool = False, **kwargs) -> Tensor:
        """Compresses the input sequence into a summary vector of size `summary_dim`.

        Parameters
        ----------
        input_sequence  : Tensor
            Input of shape (batch_size, sequence_length, input_dim)
        time            : Tensor
            Time vector of shape (batch_size, sequence_length), optional (default - None)
            Note: time values for Time2Vec embeddings will be inferred on a linearly spaced
            interval between [0, sequence length], if no time vector is specified.
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

        # Concatenate learnable time embedding to input sequence
        inp = self.time2vec(input_sequence, t=time)

        # Apply self-attention blocks
        for layer in self.attention_blocks:
            inp = layer(inp, inp, training=training, **kwargs)

        # Global average pooling and output projection
        summary = self.pooling(inp)
        summary = self.output_projector(summary)
        return summary
