import keras

from bayesflow.types import Tensor
from bayesflow.utils import layer_kwargs
from bayesflow.utils.serialization import serializable, serialize, deserialize

from .multihead_attention import MultiHeadAttention


@serializable("bayesflow.networks")
class SetAttention(keras.Layer):
    """Implements the SAB block from [1] which represents learnable self-attention.

    Wraps a :class:`MultiHeadAttention` block and calls it with ``x`` as both
    query and key/value, producing a clean one-input interface.

    [1] Lee, J., Lee, Y., Kim, J., Kosiorek, A., Choi, S., & Teh, Y. W. (2019).
        Set transformer: A framework for attention-based permutation-invariant neural networks.
        In International conference on machine learning (pp. 3744-3753). PMLR.
    """

    def __init__(
        self,
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
        """
        Parameters
        ----------
        embed_dim : int, optional
            Dimensionality of the embedding space, by default 64.
        num_heads : int, optional
            Number of attention heads, by default 4.
        dropout : float, optional
            Dropout rate applied inside the attention and FFN sublayers, by default 0.05.
        expansion_factor : float, optional
            FFN intermediate width multiplier (before the 2/3 GLU correction), by default 4.0.
        glu_variant : str, optional
            GLU activation variant for the FFN. One of ``"swiglu"``, ``"geglu"``,
            ``"reglu"``, or ``"liglu"``, by default ``"swiglu"``.
        kernel_initializer : str, optional
            Initializer for kernel weights in all dense layers, by default ``"glorot_uniform"``.
        use_bias : bool, optional
            Whether to include bias terms in dense layers, by default False.
        layer_norm : bool, optional
            Whether to apply Pre-LN RMSNorm before each sublayer, by default True.
        **kwargs
            Additional keyword arguments passed to ``keras.Layer``.
        """
        super().__init__(**layer_kwargs(kwargs))

        self.embed_dim = embed_dim
        self.num_heads = num_heads
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

    def call(self, x: Tensor, training: bool = False, attention_mask: Tensor = None) -> Tensor:
        """Performs the forward pass through the self-attention layer.

        Parameters
        ----------
        x : Tensor
            Input of shape ``(batch_size, set_size, input_dim)``.
        training : bool, optional
            Passed to dropout and norm layers, by default False.
        attention_mask : Tensor, optional
            Boolean mask of shape ``(batch_size, set_size, set_size)`` where
            1 = attend, 0 = mask.

        Returns
        -------
        Tensor
            Output of shape ``(batch_size, set_size, embed_dim)``.
        """
        return self.mab(x, x, training=training, attention_mask=attention_mask)

    def build(self, input_shape):
        self.mab.build(input_shape, input_shape)

    def compute_output_shape(self, input_shape):
        return self.mab.compute_output_shape(input_shape, input_shape)

    def get_config(self) -> dict:
        base_config = super().get_config()
        return base_config | serialize(
            {
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
