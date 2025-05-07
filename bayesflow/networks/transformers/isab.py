import keras

from bayesflow.types import Tensor
from bayesflow.utils import layer_kwargs
from bayesflow.utils.serialization import serializable

from .mab import MultiHeadAttentionBlock


@serializable("bayesflow.networks")
class InducedSetAttentionBlock(keras.Layer):
    """Implements the ISAB block from [1] which represents learnable self-attention specifically
    designed to deal with large sets via a learnable set of "inducing points".

    [1] Lee, J., Lee, Y., Kim, J., Kosiorek, A., Choi, S., & Teh, Y. W. (2019).
        Set transformer: A framework for attention-based permutation-invariant neural networks.
        In International conference on machine learning (pp. 3744-3753). PMLR.
    """

    def __init__(
        self,
        num_inducing_points: int,
        embed_dim: int = 64,
        num_heads: int = 4,
        dropout: float = 0.05,
        mlp_depth: int = 2,
        mlp_width: int = 128,
        mlp_activation: str = "gelu",
        kernel_initializer: str = "lecun_normal",
        use_bias: bool = True,
        layer_norm: bool = True,
        **kwargs,
    ):
        """Creates a self-attention attention block with inducing points (ISAB) which will typically
        be used as part of a set transformer architecture according to [1].

        Parameters
        ----------
        num_inducing_points : int, optional
            The number of inducing points for set-based dimensionality reduction.
        embed_dim : int, optional
            Dimensionality of the embedding space, by default 64.
        num_heads : int, optional
            Number of attention heads, by default 4.
        dropout : float, optional
            Dropout rate applied to attention and MLP layers, by default 0.05.
        mlp_depth : int, optional
            Number of layers in the feedforward MLP block, by default 2.
        mlp_width : int, optional
            Width of each hidden layer in the MLP block, by default 128.
        mlp_activation : str, optional
            Activation function used in the MLP block, by default "gelu".
        kernel_initializer : str, optional
            Initializer for kernel weights, by default "lecun_normal".
        use_bias : bool, optional
            Whether to include bias terms in dense layers, by default True.
        layer_norm : bool, optional
            Whether to apply layer normalization before and after attention, by default True.
        **kwargs : dict
            Additional keyword arguments passed to the Keras Layer base class.
        """

        super().__init__(**layer_kwargs(kwargs))

        self.num_inducing_points = num_inducing_points
        self.inducing_points = self.add_weight(
            shape=(self.num_inducing_points, embed_dim),
            initializer="glorot_uniform",
            trainable=True,
        )
        mab_kwargs = dict(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            mlp_depth=mlp_depth,
            mlp_width=mlp_width,
            mlp_activation=mlp_activation,
            kernel_initializer=kernel_initializer,
            use_bias=use_bias,
            layer_norm=layer_norm,
        )
        self.mab0 = MultiHeadAttentionBlock(**mab_kwargs)
        self.mab1 = MultiHeadAttentionBlock(**mab_kwargs)

    def call(self, input_set: Tensor, training: bool = False, **kwargs) -> Tensor:
        """Performs the forward pass through the self-attention layer.

        Parameters
        ----------
        input_set  : Tensor (e.g., np.ndarray, tf.Tensor, ...)
            Input of shape (batch_size, set_size, input_dim)
            Since this is self-attention, the input set is used
            as a query (Q), key (K), and value (V)
        training   : boolean, optional (default - True)
            Passed to the optional internal dropout and spectral normalization
            layers to distinguish between train and test time behavior.
        **kwargs   : dict, optional (default - {})
            Additional keyword arguments passed to the internal attention layer,
            such as ``attention_mask`` or ``return_attention_scores``

        Returns
        -------
        out : Tensor
            Output of shape (batch_size, set_size, input_dim)
        """

        batch_size = keras.ops.shape(input_set)[0]
        inducing_points_expanded = keras.ops.expand_dims(self.inducing_points, axis=0)
        inducing_points_tiled = keras.ops.tile(inducing_points_expanded, [batch_size, 1, 1])
        h = self.mab0(inducing_points_tiled, input_set, training=training, **kwargs)
        return self.mab1(input_set, h, training=training, **kwargs)
