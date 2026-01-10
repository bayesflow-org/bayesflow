from typing import Literal, Callable, Sequence

import keras

from bayesflow.networks.embeddings import FourierEmbedding
from bayesflow.networks.residual import ConditionalResidual
from bayesflow.types import Tensor
from bayesflow.utils import concatenate_valid, layer_kwargs
from bayesflow.utils.serialization import serialize, serializable, deserialize


@serializable("bayesflow.networks")
class TimeMLP(keras.Layer):
    """
    Implements a time-conditioned multi-layer perceptron (MLP).

    The model processes three separate inputs: the state variable `x`, a scalar or vector-valued time input `t`,
    and a conditioning variable `conditions`. The input and condition are projected into a shared feature space,
    merged, and passed through a deep residual MLP. A learned time embedding is injected at every hidden layer.

    If `residual` is enabled, each layer includes a skip connection for improved gradient flow. The model also
        supports dropout for regularization and spectral normalization for stability in learning smooth functions.
    """

    def __init__(
        self,
        widths: Sequence[int] = (256, 256),
        *,
        time_embedding_dim: int = 32,
        time_emb: keras.Layer | None = None,
        activation: str | Callable[[], keras.Layer] = "mish",
        kernel_initializer: str | keras.Initializer = "he_normal",
        residual: bool = True,
        dropout: Literal[0, None] | float = 0.05,
        norm: Literal["batch", "layer"] | keras.Layer = None,
        spectral_normalization: bool = False,
        merge: Literal["add", "concat"] = "concat",
        **kwargs,
    ):
        """
        Implements a time-conditioned multi-layer perceptron (MLP).

        Parameters
        ----------
        widths : Sequence[int], optional
            Defines the number of hidden units per layer, as well as the number of layers to be used.
        time_emb_dim : int
            Dimensionality of the learned time embedding. Default is 32.
        time_emb : keras.layers.Layer or None, optional
            Custom time embedding layer. If None, a random Fourier feature embedding is used.
        activation : str, optional
            Activation function applied in the hidden layers, such as "mish". Default is "mish".
        kernel_initializer : str, optional
            Initialization strategy for kernel weights, such as "he_normal". Default is "he_normal".
        residual : bool, optional
            Whether to use residual connections for improved training stability. Default is True.
        dropout : float or None, optional
            Dropout rate applied within the MLP layers for regularization. Default is 0.05.
        norm : str or keras.layers.Layer or None, optional
            Normalization applied after each hidden layer ("batch", "layer", or None). Default is "layer".
        spectral_normalization : bool, optional
            Whether to apply spectral normalization to dense layers. Default is False.
        merge : str, optional
            Method to merge input and condition if available ("add" or "concat"). Default is "concat".
        **kwargs
            Additional keyword arguments passed to `keras.Model`.
        """
        super().__init__(**kwargs)
        self.widths = list(widths)
        self.time_embedding_dim = int(time_embedding_dim)
        self.activation = activation
        self.kernel_initializer = kernel_initializer
        self.residual = residual
        self.dropout = dropout
        self.norm = norm
        self.spectral_normalization = spectral_normalization
        self.merge = merge

        if len(self.widths) == 0:
            raise ValueError("TimeMLP requires at least one hidden width.")

        # Time embedding
        if time_emb is None:
            self.time_emb = FourierEmbedding(
                embed_dim=self.time_embedding_dim,
                scale=kwargs.get("fourier_scale", 30.0),
                include_identity=True,
            )
        else:
            self.time_emb = time_emb
        self.time_proj = keras.layers.Dense(
            self.widths[0], kernel_initializer=self.kernel_initializer, name="time_proj"
        )

        # Projections for x and conditions into a shared space
        self.x_proj = keras.layers.Dense(self.widths[0] // 2, kernel_initializer=self.kernel_initializer, name="x_proj")
        self.c_proj = None
        if merge != "add" and merge != "concat":
            raise ValueError(f"Unknown merge mode: {merge!r} (expected 'add' or 'concat').")

        self.blocks = [
            ConditionalResidual(
                w,
                activation=activation,
                kernel_initializer=kernel_initializer,
                residual=residual,
                dropout=dropout,
                norm=norm,
                spectral_normalization=spectral_normalization,
                name=f"cond_block_{i}",
                **kwargs,
            )
            for i, w in enumerate(self.widths)
        ]

    @classmethod
    def from_config(cls, config, custom_objects=None):
        return cls(**deserialize(config, custom_objects=custom_objects))

    def get_config(self):
        base = super().get_config()
        base = layer_kwargs(base)
        cfg = {
            "widths": self.widths,
            "time_embedding_dim": self.time_embedding_dim,
            "time_emb": self.time_emb,
            "activation": self.activation,
            "kernel_initializer": self.kernel_initializer,
            "residual": self.residual,
            "dropout": self.dropout,
            "norm": self.norm,
            "spectral_normalization": self.spectral_normalization,
            "merge": self.merge,
        }
        return base | serialize(cfg)

    def build(self, input_shape):
        if self.built:
            return

        x_shape, t_shape, conditions_shape = input_shape
        self.time_emb.build(t_shape)
        t_emb_shape = self.time_emb.compute_output_shape(t_shape)
        self.time_proj.build(t_emb_shape)

        # Merge / input pathway
        self.x_proj.build(x_shape)
        h_shape = self.x_proj.compute_output_shape(x_shape)
        if conditions_shape is not None:
            self.c_proj = keras.layers.Dense(
                self.widths[0] // 2, kernel_initializer=self.kernel_initializer, name="c_proj"
            )
            self.c_proj.build(conditions_shape)
            if self.merge == "concat":
                h_shape = list(h_shape)
                h_shape[-1] = h_shape[-1] + self.widths[0] // 2
                h_shape = tuple(h_shape)

        # Conditional residual blocks
        for block in self.blocks:
            block.build((h_shape, t_emb_shape))
            h_shape = block.compute_output_shape((h_shape, t_emb_shape))

        super().build(input_shape)

    def compute_output_shape(self, input_shape):
        x_shape, t_shape, conditions_shape = input_shape
        h_shape = self.x_proj.compute_output_shape(x_shape)
        if conditions_shape is not None:
            if self.merge == "concat":
                h_shape = list(h_shape)
                h_shape[-1] = h_shape[-1] + self.widths[0] // 2
                h_shape = tuple(h_shape)

        t_emb_shape = self.time_emb.compute_output_shape(t_shape)

        for block in self.blocks:
            h_shape = block.compute_output_shape((h_shape, t_emb_shape))

        return h_shape

    def call(
        self,
        inputs: tuple[Tensor, Tensor, Tensor] | tuple[Tensor, Tensor, None],
        training: bool = None,
        mask=None,
    ) -> Tensor:
        x, t, conditions = inputs
        h = self.x_proj(x)
        if conditions is not None:
            hc = self.c_proj(conditions)
            if self.merge == "concat":
                h = concatenate_valid([h, hc], axis=-1)
            else:
                h = h + hc

        if keras.ops.shape(t) == 1:
            t = keras.ops.expand_dims(t, axis=-1)
        t_emb = self.time_emb(t)
        t_emb = self.time_proj(t_emb)

        for block in self.blocks:
            h = block((h, t_emb), training=training)
        return h
