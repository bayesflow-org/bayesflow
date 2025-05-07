import keras

from bayesflow.types import Tensor
from bayesflow.utils import check_lengths_same
from bayesflow.utils.serialization import serializable

from ..summary_network import SummaryNetwork

from .sab import SetAttentionBlock
from .isab import InducedSetAttentionBlock
from .pma import PoolingByMultiHeadAttention


@serializable("bayesflow.networks")
class SetTransformer(SummaryNetwork):
    """(SN) Implements the set transformer architecture from [1] which ultimately represents
    a learnable permutation-invariant function. Designed to naturally model interactions in
    the input set, which may be hard to capture with the simpler ``DeepSet`` architecture.

    [1] Lee, J., Lee, Y., Kim, J., Kosiorek, A., Choi, S., & Teh, Y. W. (2019).
        Set transformer: A framework for attention-based permutation-invariant neural networks.
        In International conference on machine learning (pp. 3744-3753). PMLR.

    Note: Currently works only on 3D inputs but can easily be expanded by using ``keras.layers.TimeDistributed``.
    """

    def __init__(
        self,
        summary_dim: int = 16,
        embed_dims: tuple = (64, 64),
        num_heads: tuple = (4, 4),
        mlp_depths: tuple = (2, 2),
        mlp_widths: tuple = (128, 128),
        num_seeds: int = 1,
        dropout: float = 0.05,
        mlp_activation: str = "gelu",
        kernel_initializer: str = "lecun_normal",
        use_bias: bool = True,
        layer_norm: bool = True,
        num_inducing_points: int = None,
        seed_dim: int = None,
        **kwargs,
    ):
        """
        Creates a many-to-one permutation-invariant encoder, typically used as a summary net for embedding set-based,
        (i.e., exchangeable or IID) data. Use a TimeSeriesTransformer or a FusionTransformer for non-IID data.

        The number of multi-head attention block is inferred from the length of `embed_dims` tuple.

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
        num_seeds   : int, optional (default - 1)
            Number of seeds to use for embedding.
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
        num_inducing_points : int or None, optional (default - None)
            Number of inducing points used, if applicable. If set to None, this option is disabled.
        seed_dim : int or None, optional (default - None)
            Dimensionality of the seed embeddings. If None, it defaults to `summary_dim`.
        **kwargs : dict
            Additional keyword arguments passed to the base layer.
        """

        super().__init__(**kwargs)

        check_lengths_same(embed_dims, num_heads, mlp_depths, mlp_widths)

        num_attention_layers = len(embed_dims)

        # Construct a series of set-attention blocks
        self.attention_blocks = keras.Sequential()

        global_attention_settings = dict(
            dropout=dropout,
            mlp_activation=mlp_activation,
            kernel_initializer=kernel_initializer,
            use_bias=use_bias,
            layer_norm=layer_norm,
        )

        for i in range(num_attention_layers):
            layer_attention_settings = dict(
                num_heads=num_heads[i],
                embed_dim=embed_dims[i],
                mlp_depth=mlp_depths[i],
                mlp_width=mlp_widths[i],
            )

            if num_inducing_points is None:
                block = SetAttentionBlock(**(global_attention_settings | layer_attention_settings))
            else:
                isab_settings = dict(num_inducing_points=num_inducing_points)
                block = InducedSetAttentionBlock(
                    **(global_attention_settings | layer_attention_settings | isab_settings)
                )

            self.attention_blocks.add(block)

        # Pooling will be applied as a final step to the abstract representations obtained from set attention
        pooling_settings = dict(
            num_heads=num_heads[-1],
            embed_dim=embed_dims[-1],
            mlp_depth=mlp_depths[-1],
            mlp_width=mlp_widths[-1],
            seed_dim=seed_dim,
            num_seeds=num_seeds,
        )
        self.pooling_by_attention = PoolingByMultiHeadAttention(**(global_attention_settings | pooling_settings))
        self.output_projector = keras.layers.Dense(summary_dim)

        self.summary_dim = summary_dim

    def call(self, input_set: Tensor, training: bool = False, **kwargs) -> Tensor:
        """Compresses the input sequence into a summary vector of size `summary_dim`.

        Parameters
        ----------
        input_set  : Tensor (e.g., np.ndarray, tf.Tensor, ...)
            Input of shape (batch_size, set_size, input_dim)
        training   : boolean, optional (default - False)
            Passed to the optional internal dropout and spectral normalization
            layers to distinguish between train and test time behavior.
        **kwargs   : dict, optional (default - {})
            Additional keyword arguments passed to the internal attention layer,
            such as ``attention_mask`` or ``return_attention_scores``

        Returns
        -------
        out : Tensor
            Output of shape (batch_size, set_size, output_dim)
        """
        summary = self.attention_blocks(input_set, training=training, **kwargs)
        summary = self.pooling_by_attention(summary, training=training, **kwargs)
        summary = self.output_projector(summary)
        return summary
