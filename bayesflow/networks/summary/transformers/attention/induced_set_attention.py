import keras

from bayesflow.types import Tensor
from bayesflow.utils import layer_kwargs
from bayesflow.utils.serialization import serializable, serialize, deserialize

from .multihead_attention import MultiHeadAttention


@serializable("bayesflow.networks")
class InducedSetAttention(keras.Layer):
    """Implements the ISAB block from [1] which represents learnable self-attention specifically
    designed to deal with large sets via a learnable set of inducing points.

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
        expansion_factor: float = 4.0,
        glu_variant: str = "swiglu",
        kernel_initializer: str = "glorot_uniform",
        use_bias: bool = False,
        layer_norm: bool = True,
        **kwargs,
    ):
        """Creates a self-attention block with inducing points (ISAB).

        Parameters
        ----------
        num_inducing_points : int
            The number of inducing points for set-based dimensionality reduction.
        embed_dim : int, optional
            Dimensionality of the embedding space, by default 64.
        num_heads : int, optional
            Number of attention heads, by default 4.
        dropout : float, optional
            Dropout rate applied inside the attention sublayer, by default 0.05.
        expansion_factor : float, optional
            FFN intermediate width multiplier (before the 2/3 GLU correction), by default 4.0.
        glu_variant : str, optional
            GLU activation variant for the FFN. One of ``"swiglu"``, ``"geglu"``,
            ``"reglu"``, or ``"liglu"``, by default ``"swiglu"``.
        kernel_initializer : str, optional
            Initializer for kernel weights, by default ``"glorot_uniform"``.
        use_bias : bool, optional
            Whether to include bias terms in dense layers, by default False.
        layer_norm : bool, optional
            Whether to apply Pre-LN RMSNorm before each sublayer, by default True.
        **kwargs
            Additional keyword arguments passed to the Keras Layer base class.
        """

        super().__init__(**layer_kwargs(kwargs))

        self.num_inducing_points = num_inducing_points
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout_rate = dropout
        self.expansion_factor = expansion_factor
        self.glu_variant = glu_variant
        self.kernel_initializer = kernel_initializer
        self.use_bias = use_bias
        self.layer_norm = layer_norm

        self.inducing_points = self.add_weight(
            shape=(self.num_inducing_points, embed_dim),
            initializer="glorot_uniform",
            trainable=True,
        )

        mab_kwargs = dict(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            expansion_factor=expansion_factor,
            glu_variant=glu_variant,
            kernel_initializer=kernel_initializer,
            use_bias=use_bias,
            layer_norm=layer_norm,
        )
        self.mab0 = MultiHeadAttention(**mab_kwargs)
        self.mab1 = MultiHeadAttention(**mab_kwargs)

    def call(self, x: Tensor, training: bool = False, attention_mask: Tensor | None = None) -> Tensor:
        """Performs the forward pass through the ISAB block.

        Parameters
        ----------
        x : Tensor
            Input of shape ``(batch_size, set_size, input_dim)``.
        training : bool, optional
            Passed to dropout and norm layers, by default False.
        attention_mask : Tensor, optional
            Boolean mask broadcastable to
            ``(batch_size, num_heads, num_inducing_points, set_size)`` where
            1 = attend, 0 = mask.

        Returns
        -------
        Tensor
            Output of shape ``(batch_size, set_size, embed_dim)``.
        """
        batch_size = keras.ops.shape(x)[0]
        inducing_points_tiled = keras.ops.tile(keras.ops.expand_dims(self.inducing_points, axis=0), [batch_size, 1, 1])
        h = self.mab0(inducing_points_tiled, x, training=training, attention_mask=attention_mask)
        return self.mab1(x, h, training=training)

    def build(self, input_shape):
        if self.built:
            return

        input_shape = input_shape
        inducing_shape = (input_shape[0], self.num_inducing_points, self.embed_dim)
        self.mab0.build(inducing_shape, input_shape)
        h_shape = self.mab0.compute_output_shape(inducing_shape, input_shape)
        self.mab1.build(input_shape, h_shape)

    def compute_output_shape(self, input_shape):
        return tuple(input_shape)[:-1] + (self.embed_dim,)

    def get_config(self) -> dict:
        base_config = super().get_config()
        return base_config | serialize(
            {
                "num_inducing_points": self.num_inducing_points,
                "embed_dim": self.embed_dim,
                "num_heads": self.num_heads,
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
