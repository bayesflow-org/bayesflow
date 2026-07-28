import keras

from bayesflow.types import Tensor
from bayesflow.utils import layer_kwargs
from bayesflow.utils.serialization import serializable, serialize, deserialize

from bayesflow.networks.helpers import FFN
from .multihead_attention import MultiHeadAttention


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
        expansion_factor: float = 4.0,
        glu_variant: str = "swiglu",
        kernel_initializer: str = "glorot_uniform",
        use_bias: bool = False,
        layer_norm: bool = True,
        **kwargs,
    ):
        """
        Creates a PoolingByMultiHeadAttention (PMA) block for permutation-invariant set encoding using
        multi-head attention pooling. Can also be used as a building block for ``DeepSet`` architectures.

        Parameters
        ----------
        num_seeds : int, optional
            Number of seed vectors used for pooling. Acts as the number of summary outputs,
            by default 1.
        embed_dim : int, optional
            Dimensionality of the embedding space used in the attention mechanism, by default 64.
        num_heads : int, optional
            Number of attention heads in the multi-head attention block, by default 4.
        seed_dim : int or None, optional
            Dimensionality of each seed vector. If None, defaults to ``embed_dim``.
        dropout : float, optional
            Dropout rate applied inside the attention sublayer, by default 0.05.
        expansion_factor : float, optional
            FFN intermediate width multiplier (before the 2/3 GLU correction), by default 4.0.
        glu_variant : str, optional
            GLU activation variant for both the pre-attention FFN and the MAB's internal FFN.
            One of ``"swiglu"``, ``"geglu"``, ``"reglu"``, or ``"liglu"``, by default ``"swiglu"``.
        kernel_initializer : str, optional
            Initializer for kernel weights in all dense layers, by default ``"glorot_uniform"``.
        use_bias : bool, optional
            Whether to include bias terms in dense layers, by default False.
        layer_norm : bool, optional
            Whether to apply Pre-LN RMSNorm before each sublayer, by default True.
        **kwargs
            Additional keyword arguments passed to the Keras Layer base class.
        """

        super().__init__(**layer_kwargs(kwargs))

        self.num_seeds = num_seeds
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.seed_dim = seed_dim
        self.dropout_rate = dropout
        self.expansion_factor = expansion_factor
        self.glu_variant = glu_variant
        self.kernel_initializer = kernel_initializer
        self.use_bias = use_bias
        self.layer_norm = layer_norm

        self.mab = MultiHeadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            expansion_factor=expansion_factor,
            glu_variant=glu_variant,
            kernel_initializer=kernel_initializer,
            use_bias=use_bias,
            layer_norm=layer_norm,
        )

        self.seed_vector = self.add_weight(
            shape=(num_seeds, seed_dim if seed_dim is not None else embed_dim),
            initializer="glorot_uniform",
            trainable=True,
        )

        # Pre-attention FFN: refines set representations before pooling (rFF in the paper)
        self.feedforward = FFN(
            embed_dim=embed_dim,
            expansion_factor=expansion_factor,
            glu_variant=glu_variant,
            use_bias=use_bias,
            dropout=dropout,
            kernel_initializer=kernel_initializer,
        )

    def call(
        self,
        x: Tensor,
        training: bool = False,
        attention_mask: Tensor | None = None,
    ) -> Tensor:
        """Performs the forward pass through the PMA block.

        Parameters
        ----------
        x : Tensor
            Input of shape ``(batch_size, set_size, input_dim)``.
        training : bool, optional
            Passed to dropout and norm layers, by default False.
        attention_mask : Tensor, optional
            Boolean mask broadcastable to
            ``(batch_size, num_heads, num_seeds, set_size)`` where 1 = attend,
            0 = mask.

        Returns
        -------
        Tensor
            Output of shape ``(batch_size, num_seeds * embed_dim)``.
        """
        set_x_transformed = self.feedforward(x, training=training)
        batch_size = keras.ops.shape(x)[0]
        seed_tiled = keras.ops.tile(keras.ops.expand_dims(self.seed_vector, axis=0), [batch_size, 1, 1])
        summaries = self.mab(
            seed_tiled,
            set_x_transformed,
            training=training,
            attention_mask=attention_mask,
        )
        return keras.ops.reshape(summaries, (keras.ops.shape(summaries)[0], -1))

    def build(self, input_shape):
        if self.built:
            return

        input_shape = input_shape
        self.feedforward.build(input_shape)
        ffn_out_shape = self.feedforward.compute_output_shape(input_shape)
        seed_shape = (input_shape[0], self.num_seeds, self.seed_dim if self.seed_dim is not None else self.embed_dim)
        self.mab.build(seed_shape, ffn_out_shape)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.num_seeds * self.embed_dim)

    def get_config(self):
        base_config = super().get_config()
        return base_config | serialize(
            {
                "num_seeds": self.num_seeds,
                "embed_dim": self.embed_dim,
                "num_heads": self.num_heads,
                "seed_dim": self.seed_dim,
                "dropout": self.dropout_rate,
                "expansion_factor": self.expansion_factor,
                "glu_variant": self.glu_variant,
                "kernel_initializer": self.kernel_initializer,
                "use_bias": self.use_bias,
                "layer_norm": self.layer_norm,
            }
        )

    @classmethod
    def from_config(cls, config, custom_objects=None):
        return cls(**deserialize(config, custom_objects=custom_objects))
