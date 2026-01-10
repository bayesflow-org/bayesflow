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
    merged, and passed through a deep residual MLP. A learned time embedding is injected via FiLM at every
    hidden layer.

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

        self.merge_proj = None
        if merge == "add":
            first_width = self.widths[0]
            # Projections for x and conditions into a shared space
            self.x_proj = keras.layers.Dense(first_width, kernel_initializer=kernel_initializer, name="x_proj")
            self.c_proj = keras.layers.Dense(first_width, kernel_initializer=kernel_initializer, name="c_proj")
        elif merge == "concat":
            self.x_proj = None
            self.c_proj = None
        else:
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

    def build(self, x_shape, t_shape, conditions_shape=None):
        if self.built:
            return

        # Time embedding
        t_emb_shape = self.time_embedding_dim + 1  # include_identity=True adds 1

        # Merge / input pathway
        if self.merge == "add" and conditions_shape is not None:
            self.x_proj.build(x_shape)
            self.c_proj.build(conditions_shape)
            h_shape = self.x_proj.compute_output_shape(x_shape)
        else:
            h_shape = x_shape
            if conditions_shape is not None:
                h_shape = list(h_shape)
                h_shape[-1] += int(conditions_shape[-1])
                h_shape = tuple(h_shape)

        # Conditional residual blocks
        for block in self.blocks:
            block.build((h_shape, t_emb_shape))
            h_shape = block.compute_output_shape((h_shape, t_emb_shape))

    def compute_output_shape(self, x_shape, t_shape, conditions_shape=None):
        if self.merge == "add" and conditions_shape is not None:
            h_shape = self.x_proj.compute_output_shape(x_shape)
        else:
            h_shape = x_shape
            if conditions_shape is not None:
                h_shape = list(h_shape)
                h_shape[-1] += int(conditions_shape[-1])
                h_shape = tuple(h_shape)

        t_emb_shape = self.time_embedding_dim + 1  # include_identity=True adds 1

        for block in self.blocks:
            h_shape = block.compute_output_shape((h_shape, t_emb_shape))

        return h_shape

    def call(
        self, x: Tensor, t: Tensor, conditions: Tensor | None = None, training: bool = None, mask: bool = None
    ) -> Tensor:
        if conditions is None:
            h = x
        else:
            if self.merge == "add":
                hx = self.x_proj(x)
                hc = self.c_proj(conditions)
                h = hx + hc
            else:
                h = concatenate_valid([x, conditions], axis=-1)

        t_emb = self.time_emb(t)

        for block in self.blocks:
            h = block((h, t_emb), training=training, mask=mask)

        return h
