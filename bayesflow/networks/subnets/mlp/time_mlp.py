from typing import Literal, Callable, Sequence

import keras

from bayesflow.types import Tensor
from bayesflow.utils import layer_kwargs
from bayesflow.utils.serialization import serialize, serializable, deserialize

from ...helpers import FourierEmbedding
from ...helpers import ConditionalBlock


@serializable("bayesflow.networks")
class TimeMLP(keras.Layer):
    """Time-conditioned multi-layer perceptron with FiLM modulation.

    Processes three inputs: a state variable ``x``, a scalar or vector-valued
    time ``t``, and an optional conditioning variable ``conditions``.  The input
    and conditions are projected into a shared feature space, merged, and passed
    through residual blocks.  A learned time embedding is injected via FiLM at
    every hidden layer.

    Parameters
    ----------
    widths : Sequence[int], optional
        Number of hidden units per layer. Default is ``(256, 256)``.
    time_embedding_dim : int, optional
        Dimensionality of the learned time embedding. Default is ``32``.
        Set to ``1`` to use time directly without embedding.
    time_emb : keras.Layer or None, optional
        Custom time embedding layer. If ``None``, uses random Fourier features.
    fourier_scale : float, optional
        Frequency scaling for the default Fourier embedding. Default is ``30.0``.
        Ignored when *time_emb* is provided.
    activation : str or callable, optional
        Activation function for hidden layers. Default is ``"mish"``.
    kernel_initializer : str or keras.Initializer, optional
        Weight initialization strategy. Default is ``"he_normal"``.
    residual : bool, optional
        Whether to use residual connections. Default is ``True``.
    dropout : float or None, optional
        Dropout rate for regularization. Default is ``0.05``.
    norm : ``"batch"``, ``"layer"``, keras.Layer, or None, optional
        Normalization applied after each hidden layer. Default is ``"layer"``.
    spectral_normalization : bool, optional
        Whether to apply spectral normalization to Dense layers.
        Default is ``False``.
    merge : ``"add"`` or ``"concat"``, optional
        How to merge input and conditions (``"add"`` or ``"concat"``).
        Default is ``"concat"``.
    **kwargs
        Additional keyword arguments passed to ``keras.Layer``.
    """

    def __init__(
        self,
        widths: Sequence[int] = (256, 256),
        *,
        time_embedding_dim: int = 32,
        time_emb: keras.Layer | None = None,
        fourier_scale: float = 30.0,
        activation: str | Callable[[], keras.Layer] = "mish",
        kernel_initializer: str | keras.Initializer = "he_normal",
        residual: bool = True,
        dropout: Literal[0, None] | float = 0.05,
        norm: Literal["batch", "layer"] | keras.Layer = "layer",
        spectral_normalization: bool = False,
        merge: Literal["add", "concat"] = "concat",
        **kwargs,
    ):
        super().__init__(**layer_kwargs(kwargs))

        if len(widths) == 0:
            raise ValueError("TimeMLP requires at least one hidden width.")
        if merge not in ("add", "concat"):
            raise ValueError(f"Unknown merge mode: {merge!r} (expected 'add' or 'concat').")

        self.widths = list(widths)
        self.time_embedding_dim = time_embedding_dim
        self.fourier_scale = fourier_scale
        self.activation = activation
        self.kernel_initializer = kernel_initializer
        self.residual = residual
        self.dropout = dropout
        self.norm = norm
        self.spectral_normalization = spectral_normalization
        self.merge = merge

        # Time embedding
        if time_emb is None:
            if self.time_embedding_dim == 1:
                self.time_emb = keras.layers.Identity()
            else:
                self.time_emb = FourierEmbedding(
                    embed_dim=self.time_embedding_dim,
                    scale=self.fourier_scale,
                    include_identity=True,
                )
        else:
            self.time_emb = time_emb

        # Projections for x and conditions into a shared space
        self.x_proj = keras.layers.Dense(self.widths[0], kernel_initializer=self.kernel_initializer, name="x_proj")
        self.c_proj = None
        self.merge_proj = None

        act = keras.activations.get(activation)
        if not isinstance(act, keras.Layer):
            act = keras.layers.Activation(act)
        self.act = act

        # Time-conditional blocks
        self.blocks = [
            ConditionalBlock(
                width=width,
                activation=activation,
                kernel_initializer=kernel_initializer,
                residual=residual,
                dropout=dropout,
                norm=norm,
                spectral_normalization=spectral_normalization,
            )
            for width in self.widths
        ]

    @classmethod
    def from_config(cls, config, custom_objects=None):
        return cls(**deserialize(config, custom_objects=custom_objects))

    def get_config(self):
        base_config = super().get_config()
        base_config = layer_kwargs(base_config)

        config = {
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
        return base_config | serialize(config)

    def build(self, input_shape):
        if self.built:
            return

        x_shape, t_shape, conditions_shape = input_shape

        # Time embedding
        self.time_emb.build(t_shape)
        t_emb_shape = self.time_emb.compute_output_shape(t_shape)

        # Input projection
        self.x_proj.build(x_shape)
        h_shape = self.x_proj.compute_output_shape(x_shape)

        # Condition projection and merge
        if conditions_shape is not None:
            self.c_proj = keras.layers.Dense(self.widths[0], kernel_initializer=self.kernel_initializer, name="c_proj")
            self.c_proj.build(conditions_shape)

            if self.merge == "concat":
                merge_shape = list(h_shape)
                merge_shape[-1] = merge_shape[-1] + self.widths[0]
                merge_shape = tuple(merge_shape)
            else:
                merge_shape = h_shape

            self.merge_proj = keras.layers.Dense(
                self.widths[0], kernel_initializer=self.kernel_initializer, name="merge_proj"
            )
            self.merge_proj.build(merge_shape)

        # Time-conditional blocks
        for block in self.blocks:
            block.build((h_shape, t_emb_shape))
            h_shape = block.compute_output_shape((h_shape, t_emb_shape))

        super().build(input_shape)

    def compute_output_shape(self, input_shape):
        x_shape, t_shape, conditions_shape = input_shape
        h_shape = self.x_proj.compute_output_shape(x_shape)
        t_emb_shape = self.time_emb.compute_output_shape(t_shape)

        for block in self.blocks:
            h_shape = block.compute_output_shape((h_shape, t_emb_shape))

        return h_shape

    def call(
        self,
        inputs: tuple[Tensor, Tensor, Tensor | None],
        training: bool = None,
    ) -> Tensor:
        x, t, conditions = inputs

        # Project and merge inputs
        h = self.x_proj(x)
        if conditions is not None and self.c_proj is not None:
            hc = self.c_proj(conditions)
            if self.merge == "concat":
                h = keras.ops.concatenate([h, hc], axis=-1)
            else:
                h = h + hc
            h = self.merge_proj(self.act(h))
        h = self.act(h)

        # Time embedding
        t_emb = self.time_emb(t)

        # Residual blocks with time conditioning
        for block in self.blocks:
            h = block((h, t_emb), training=training)

        return h
