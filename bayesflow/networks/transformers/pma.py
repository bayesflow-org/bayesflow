import keras
import keras.ops as ops
from keras.saving import register_keras_serializable as serializable

from bayesflow.types import Tensor
from bayesflow.networks import MLP
from .mab import MultiHeadAttentionBlock


@serializable(package="bayesflow.networks")
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
        num_dense_feedforward: int = 2,
        dense_units: int = 128,
        dense_activation: str = "gelu",
        kernel_initializer: str = "he_normal",
        use_bias: bool = True,
        layer_norm: bool = True,
        **kwargs,
    ):
        """Creates a multi-head attention block (MAB) which will perform cross-attention between an input set
        and a set of seed vectors (typically one for a single summary) with summary_dim output dimensions.

        Could also be used as part of a ``DeepSet`` for representing learnable instead of fixed pooling.

        Parameters
        ----------
        ##TODO
        """

        super().__init__(**kwargs)

        self.mab = MultiHeadAttentionBlock(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            num_dense_feedforward=num_dense_feedforward,
            dense_units=dense_units,
            dense_activation=dense_activation,
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
            depth=num_dense_feedforward,
            width=dense_units,
            activation=dense_activation,
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
