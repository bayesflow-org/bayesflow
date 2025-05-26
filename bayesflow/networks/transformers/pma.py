import keras
import keras.ops as ops

from bayesflow.networks import MLP
from bayesflow.types import Tensor
from bayesflow.utils import layer_kwargs
from bayesflow.utils.decorators import sanitize_input_shape
from bayesflow.utils.serialization import serializable

from .mab import MultiHeadAttentionBlock


@serializable("bayesflow.networks")
class PoolingByMultiHeadAttention(keras.Layer):
    """Implements the pooling with multi-head attention (PMA) block from [1] which represents
    a permutation-invariant encoder for set-based inputs.

    [1] Lee, J., Lee, Y., Kim, J., Kosiorek, A., Choi, S., & Teh, Y. W. (2019).
        Set transformer: A framework for attention-based permutation-invariant neural networks.
        In International conference on machine learning (pp. 3744-3753). PMLR.

    Note: Currently works only on 3D inputs but can easily be expanded by changing
    the internals slightly or using ``keras.layers.TimeDistributed``.
    """

    def __init__(
        self,
        num_seeds: int = 1,
        embed_dim: int = 64,
        num_heads: int = 4,
        seed_dim: int = None,
        dropout: float = 0.05,
        mlp_depth: int = 2,
        mlp_width: int = 128,
        mlp_activation: str = "gelu",
        kernel_initializer: str = "lecun_normal",
        use_bias: bool = True,
        layer_norm: bool = True,
        **kwargs,
    ):
        """
        Creates a PoolingByMultiHeadAttention (PMA) block for permutation-invariant set encoding using
        multi-head attention pooling. Can also be used us a building block for `DeepSet` architectures.

        Parameters
        ----------
        num_seeds : int, optional (default=1)
            Number of seed vectors used for pooling. Acts as the number of summary outputs.
        embed_dim : int, optional (default=64)
            Dimensionality of the embedding space used in the attention mechanism.
        num_heads : int, optional (default=4)
            Number of attention heads in the multi-head attention block.
        seed_dim : int or None, optional (default=None)
            Dimensionality of each seed vector. If None, defaults to `embed_dim`.
        dropout : float, optional (default=0.05)
            Dropout rate applied to attention and MLP layers.
        mlp_depth : int, optional (default=2)
            Number of layers in the feedforward MLP applied before attention.
        mlp_width : int, optional (default=128)
            Number of units in each hidden layer of the MLP.
        mlp_activation : str, optional (default="gelu")
            Activation function used in the MLP.
        kernel_initializer : str, optional (default="lecun_normal")
            Initializer for kernel weights in dense layers.
        use_bias : bool, optional (default=True)
            Whether to include bias terms in dense layers.
        layer_norm : bool, optional (default=True)
            Whether to apply layer normalization before and after attention.
        **kwargs
            Additional keyword arguments passed to the Keras Layer base class.
        """

        super().__init__(**layer_kwargs(kwargs))

        self.mab = MultiHeadAttentionBlock(
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

        self.seed_vector = self.add_weight(
            shape=(num_seeds, seed_dim if seed_dim is not None else embed_dim),
            initializer="glorot_uniform",
            trainable=True,
        )

        self.feedforward = MLP(
            widths=(mlp_width,) * mlp_depth,
            activation=mlp_activation,
            kernel_initializer=kernel_initializer,
            dropout=dropout,
        )

    def call(self, input_set: Tensor, training: bool = False, **kwargs) -> Tensor:
        """Performs the forward pass through the PMA block.

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
        summary : Tensor
            Output of shape (batch_size, num_seeds * summary_dim)
        """

        set_x_transformed = self.feedforward(input_set, training=training)
        batch_size = ops.shape(input_set)[0]
        seed_vector_expanded = ops.expand_dims(self.seed_vector, axis=0)
        seed_tiled = ops.tile(seed_vector_expanded, [batch_size, 1, 1])
        summaries = self.mab(seed_tiled, set_x_transformed, training=training, **kwargs)
        return ops.reshape(summaries, (ops.shape(summaries)[0], -1))

    @sanitize_input_shape
    def compute_output_shape(self, input_shape):
        return keras.ops.shape(self.call(keras.ops.zeros(input_shape)))
