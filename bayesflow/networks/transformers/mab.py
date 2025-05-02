import keras
from keras import layers

from bayesflow.networks import MLP
from bayesflow.types import Tensor
from bayesflow.utils import layer_kwargs
from bayesflow.utils.decorators import sanitize_input_shape
from bayesflow.utils.serialization import serializable


@serializable("bayesflow.networks")
class MultiHeadAttentionBlock(keras.Layer):
    """Implements the MAB block from [1] which represents learnable cross-attention.

    In particular, it uses a so-called "Post-LN" transformer block [2] which applies
    layer norm following attention and following MLP. A "Pre-LN" transformer block
    can easily be implemented.

    [1] Lee, J., Lee, Y., Kim, J., Kosiorek, A., Choi, S., & Teh, Y. W. (2019).
        Set transformer: A framework for attention-based permutation-invariant neural networks.
        In International conference on machine learning (pp. 3744-3753). PMLR.

    [2] Xiong, R., Yang, Y., He, D., Zheng, K., Zheng, S., Xing, C., ... & Liu, T. (2020, November).
    On layer normalization in the transformer architecture.
    In International conference on machine learning (pp. 10524-10533). PMLR.
    """

    def __init__(
        self,
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
        """Creates a multi-head attention block which will typically be used as part of a
        set transformer architecture according to [1]. Corresponds to standard cross-attention.

        Parameters
        ----------
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

        self.input_projector = layers.Dense(embed_dim, name="input_projector")
        self.attention = layers.MultiHeadAttention(
            key_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            use_bias=use_bias,
            output_shape=embed_dim,
            name="attention",
        )
        self.ln_pre = layers.LayerNormalization(name="layer_norm_pre") if layer_norm else None
        self.mlp = MLP(
            widths=(mlp_width,) * mlp_depth,
            activation=mlp_activation,
            kernel_initializer=kernel_initializer,
            dropout=dropout,
        )
        self.output_projector = layers.Dense(embed_dim, name="output_projector")
        self.ln_post = layers.LayerNormalization(name="layer_norm_post") if layer_norm else None

    def call(self, seq_x: Tensor, seq_y: Tensor, training: bool = False, **kwargs) -> Tensor:
        """Performs the forward pass through the attention layer.

        Parameters
        ----------
        seq_x    : Tensor (e.g., np.ndarray, tf.Tensor, ...)
            Input of shape (batch_size, seq_size_x, input_dim), which will
            play the role of a query (Q).
        seq_y    : Tensor
            Input of shape (batch_size, seq_size_y, input_dim), which will
            play the role of key (K) and value (V).
        training : boolean, optional (default - True)
            Passed to the optional internal dropout and spectral normalization
            layers to distinguish between train and test time behavior.
        **kwargs : dict, optional (default - {})
            Additional keyword arguments passed to the internal attention layer,
            such as ``attention_mask`` or ``return_attention_scores``

        Returns
        -------
        out : Tensor
            Output of shape (batch_size, set_size_x, output_dim)
        """

        h = self.input_projector(seq_x) + self.attention(
            query=seq_x, key=seq_y, value=seq_y, training=training, **kwargs
        )
        if self.ln_pre is not None:
            h = self.ln_pre(h, training=training)

        out = h + self.output_projector(self.mlp(h, training=training))
        if self.ln_post is not None:
            out = self.ln_post(out, training=training)

        return out

    # noinspection PyMethodOverriding
    @sanitize_input_shape
    def build(self, seq_x_shape, seq_y_shape):
        self.call(keras.ops.zeros(seq_x_shape), keras.ops.zeros(seq_y_shape))

    @sanitize_input_shape
    def compute_output_shape(self, seq_x_shape, seq_y_shape):
        return keras.ops.shape(self.call(keras.ops.zeros(seq_x_shape), keras.ops.zeros(seq_y_shape)))
