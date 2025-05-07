import keras
from keras import layers

from bayesflow.types import Tensor
from bayesflow.utils import check_lengths_same
from bayesflow.utils.serialization import serializable

from ..summary_network import SummaryNetwork

from .mab import MultiHeadAttentionBlock


@serializable("bayesflow.networks")
class FusionTransformer(SummaryNetwork):
    """
    (SN) Implements a more flexible version of the TimeSeriesTransformer that applies a series of self-attention layers
    followed by cross-attention between the representation and a learnable template summarized via a recurrent net."""

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
        template_type: str = "lstm",
        bidirectional: bool = True,
        template_dim: int = 128,
        **kwargs,
    ):
        """Creates a fusion transformer used to flexibly compress time series and learn additional time embeddings
        using a recurrent neural network. If the time intervals vary across batches, it is highly recommended that
        your simulator also returns a "time" vector appended to the simulator outputs.

        Important: This network needs at least 2 transformer blocks and will generally be slower than the
        corresponding TimeSeriesTransformer.

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
        t2v_embed_dim : int, optional (default - 8)
            The dimensionality of the Time2Vec embedding.
        template_type        : str or callable, optional, default: 'lstm'
            The many-to-one (learnable) transformation of the time series.
            if ``lstm``, an LSTM network will be used.
            if ``gru``, a GRU unit will be used.
        bidirectional        : bool, optional (default - False)
            Indicates whether the involved recurrent template network is bidirectional (i.e., forward
            and backward in time) or unidirectional (forward in time). Defaults to False, but may
            increase performance in some applications.
        template_dim         : int, optional (default - 128)
            Only used if ``template_type`` in ['lstm', 'gru']. The number of hidden
            units (equiv. output dimensions) of the recurrent network.
        time_axis     : int, optional (default - None)
            The time axis (e.g., -1 for last axis) from which to grab the time vector that goes into t2v.
        **kwargs : dict
            Additional keyword arguments passed to the base layer.
        """

        super().__init__(**kwargs)

        # Ensure all tuple-settings have the same length
        check_lengths_same(embed_dims, num_heads, mlp_depths, mlp_widths)

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

        # A recurrent network will learn a dynamic many-to-one template
        if template_type.upper() == "LSTM":
            self.template_net = (
                layers.Bidirectional(layers.LSTM(template_dim // 2, dropout=dropout))
                if bidirectional
                else (layers.LSTM(template_dim, dropout=dropout))
            )
        elif template_type.upper() == "GRU":
            self.template_net = (
                layers.Bidirectional(layers.GRU(template_dim // 2, dropout=dropout))
                if bidirectional
                else (layers.GRU(template_dim, dropout=dropout))
            )
        else:
            raise ValueError("Argument `template_dim` should be in ['lstm', 'gru']")

        self.output_projector = keras.layers.Dense(summary_dim)
        self.summary_dim = summary_dim

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

        template = self.template_net(input_sequence, training=training)

        rep = input_sequence
        for layer in self.attention_blocks[:-1]:
            rep = layer(rep, rep, training=training, **kwargs)

        summary = self.attention_blocks[-1](keras.ops.expand_dims(template, axis=1), rep, training=training, **kwargs)
        summary = self.output_projector(keras.ops.squeeze(summary, axis=1))
        return summary
