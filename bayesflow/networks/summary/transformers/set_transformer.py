import keras

from bayesflow.types import Tensor
from bayesflow.utils import check_lengths_same
from bayesflow.utils.serialization import serializable

from .transformer import Transformer
from .attention import SetAttention, InducedSetAttention, PoolingByMultiHeadAttention


@serializable("bayesflow.networks")
class SetTransformer(Transformer):
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
        num_seeds: int = 4,
        dropout: float = 0.05,
        expansion_factor: float = 4.0,
        glu_variant: str = "swiglu",
        kernel_initializer: str = "glorot_uniform",
        use_bias: bool = False,
        layer_norm: bool = True,
        num_inducing_points: int = None,
        seed_dim: int = None,
        **kwargs,
    ):
        """
        Creates a many-to-one permutation-invariant encoder, typically used as a summary network
        for embedding set-based (i.e., exchangeable or IID) data.

        Parameters
        ----------
        summary_dim : int, optional
            Dimensionality of the final summary output, by default 16.
        embed_dims : tuple of int, optional
            Embedding dimensionality for each attention block, by default (64, 64).
        num_heads : tuple of int, optional
            Number of attention heads for each block, by default (4, 4).
        num_seeds : int, optional
            Number of seed vectors used for PMA pooling. Increase if performance
            appears subpar. By default 4.
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
        num_inducing_points : int or None, optional
            If set, uses InducedSetAttention (ISAB) blocks with this many inducing
            points instead of standard SetAttention (SAB) blocks.
        seed_dim : int or None, optional
            Dimensionality of the PMA seed vectors. If None, defaults to ``embed_dims[-1]``.
        **kwargs
            Additional keyword arguments passed to the base layer.
        """

        super().__init__(**kwargs)

        check_lengths_same(embed_dims, num_heads)

        shared_kwargs = dict(
            dropout=dropout,
            expansion_factor=expansion_factor,
            glu_variant=glu_variant,
            kernel_initializer=kernel_initializer,
            use_bias=use_bias,
            layer_norm=layer_norm,
        )

        self.attention_blocks = []
        for i in range(len(embed_dims)):
            block_kwargs = shared_kwargs | dict(num_heads=num_heads[i], embed_dim=embed_dims[i])
            if num_inducing_points is None:
                block = SetAttention(**block_kwargs)
            else:
                block = InducedSetAttention(num_inducing_points=num_inducing_points, **block_kwargs)
            self.attention_blocks.append(block)

        self.pooling_by_attention = PoolingByMultiHeadAttention(
            num_heads=num_heads[-1],
            embed_dim=embed_dims[-1],
            num_seeds=num_seeds,
            seed_dim=seed_dim,
            **shared_kwargs,
        )
        self.output_projector = keras.layers.Dense(units=summary_dim)

        self.summary_dim = summary_dim

    def call(self, x: Tensor, training: bool = False, attention_mask: Tensor = None) -> Tensor:
        """Compresses the input set into a summary vector of size ``summary_dim``.

        Parameters
        ----------
        x : Tensor
            Input of shape ``(batch_size, set_size, input_dim)``.
        training : bool, optional
            Passed to dropout and norm layers, by default False.
        attention_mask : Tensor, optional
            Boolean mask of shape ``(B, T, T)`` where 1 = attend, 0 = mask.

        Returns
        -------
        Tensor
            Output of shape ``(batch_size, summary_dim)``.
        """
        for layer in self.attention_blocks:
            x = layer(x, training=training, attention_mask=attention_mask)
        x = self.pooling_by_attention(x, training=training)
        x = self.output_projector(x)
        return x
