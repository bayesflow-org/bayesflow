from collections.abc import Sequence
from typing import Literal, Callable

import keras

from bayesflow.utils import layer_kwargs
from bayesflow.utils.serialization import deserialize, serializable, serialize

from ...helpers import DenseBlock


def _unpack_shape(input_shape):
    """Unpack *input_shape* into ``(x_shape, conditions_shape)``.

    Accepts plain shapes ``(batch, features)`` as well as structured shapes
    ``((batch, features), (batch, cond_features))`` or
    ``((batch, features), None)``.
    """
    if isinstance(input_shape, (tuple, list)) and len(input_shape) == 2:
        if isinstance(input_shape[0], (tuple, list)):
            x_shape = tuple(input_shape[0])
            cond_shape = tuple(input_shape[1]) if input_shape[1] is not None else None
            return x_shape, cond_shape
    return tuple(input_shape) if isinstance(input_shape, list) else input_shape, None


def _unpack_inputs(inputs):
    """Unpack call-time *inputs* into ``(x, conditions)``."""
    if isinstance(inputs, (tuple, list)) and len(inputs) == 2:
        return inputs[0], inputs[1]
    return inputs, None


@serializable("bayesflow.networks")
class MLP(keras.Layer):
    """Flexible multi-layer perceptron with optional conditional input,
    residual connections, dropout, and spectral normalization.

    Supports two input modes determined at build time:

    - **Unconditional** — ``build(x_shape)`` / ``call(x)``
    - **Conditional** — ``build((x_shape, conditions_shape))`` /
      ``call((x, conditions))``

    In conditional mode the input and conditions are each projected into a
    shared feature space, merged (via concatenation or addition), and then
    passed through the hidden blocks.  Without conditions the input feeds
    directly into the blocks.

    Parameters
    ----------
    widths : Sequence[int], optional
        Number of hidden units per layer, determining both width and depth.
        Default is ``(256, 256)``.
    activation : str or callable, optional
        Activation function for hidden layers. Default is ``"mish"``.
    kernel_initializer : str or keras.Initializer, optional
        Weight initialization strategy. Default is ``"he_normal"``.
    residual : bool, optional
        Whether to use residual (skip) connections. Default is ``True``.
    dropout : float or None, optional
        Dropout rate for regularization. Default is ``0.05``.
    norm : ``"batch"``, ``"layer"``, keras.Layer, or None, optional
        Normalization applied after each hidden layer. Default is ``None``.
    spectral_normalization : bool, optional
        Apply spectral normalization to Dense layers. Default is ``False``.
    merge : ``"add"`` or ``"concat"``, optional
        How to merge input and conditions in conditional mode.
        Default is ``"concat"``.
    **kwargs
        Additional keyword arguments passed to ``keras.Layer``.
    """

    def __init__(
        self,
        widths: Sequence[int] = (256, 256),
        *,
        activation: str | Callable[[], keras.Layer] = "mish",
        kernel_initializer: str | keras.Initializer = "he_normal",
        residual: bool = True,
        dropout: Literal[0, None] | float = 0.05,
        norm: Literal["batch", "layer"] | keras.Layer = None,
        spectral_normalization: bool = False,
        merge: Literal["add", "concat"] = "concat",
        **kwargs,
    ):
        super().__init__(**layer_kwargs(kwargs))

        if len(widths) == 0:
            raise ValueError("MLP requires at least one hidden width.")
        if merge not in ("add", "concat"):
            raise ValueError(f"Unknown merge mode: {merge!r} (expected 'add' or 'concat').")

        self.widths = list(widths)
        self.activation = activation
        self.kernel_initializer = kernel_initializer
        self.residual = residual
        self.dropout = dropout
        self.norm = norm
        self.spectral_normalization = spectral_normalization
        self.merge = merge

        # Hidden blocks
        self.blocks = [
            DenseBlock(
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

        # Input pathway — populated in build() when conditions are present
        self.x_proj = None
        self.c_proj = None
        self.merge_proj = None

        act = keras.activations.get(activation)
        if not isinstance(act, keras.Layer):
            act = keras.layers.Activation(act)
        self.input_act = act

    # -- Serialization -------------------------------------------------------

    @classmethod
    def from_config(cls, config, custom_objects=None):
        return cls(**deserialize(config, custom_objects=custom_objects))

    def get_config(self):
        base_config = super().get_config()
        base_config = layer_kwargs(base_config)

        config = {
            "widths": self.widths,
            "activation": self.activation,
            "kernel_initializer": self.kernel_initializer,
            "residual": self.residual,
            "dropout": self.dropout,
            "norm": self.norm,
            "spectral_normalization": self.spectral_normalization,
            "merge": self.merge,
        }

        return base_config | serialize(config)

    # -- Build / call --------------------------------------------------------

    def build(self, input_shape):
        if self.built:
            return

        x_shape, conditions_shape = _unpack_shape(input_shape)
        h_shape = x_shape

        # Conditional pathway
        if conditions_shape is not None:
            self.x_proj = keras.layers.Dense(self.widths[0], kernel_initializer=self.kernel_initializer, name="x_proj")
            self.x_proj.build(x_shape)
            h_shape = self.x_proj.compute_output_shape(x_shape)

            self.c_proj = keras.layers.Dense(self.widths[0], kernel_initializer=self.kernel_initializer, name="c_proj")
            self.c_proj.build(conditions_shape)

            if self.merge == "concat":
                merge_shape = h_shape[:-1] + (h_shape[-1] + self.widths[0],)
            else:
                merge_shape = h_shape

            self.merge_proj = keras.layers.Dense(
                self.widths[0], kernel_initializer=self.kernel_initializer, name="merge_proj"
            )
            self.merge_proj.build(merge_shape)
            h_shape = self.merge_proj.compute_output_shape(merge_shape)

        # Hidden blocks
        for block in self.blocks:
            block.build(h_shape)
            h_shape = block.compute_output_shape(h_shape)

        super().build(input_shape)

    def compute_output_shape(self, input_shape):
        x_shape, _ = _unpack_shape(input_shape)
        return x_shape[:-1] + (self.widths[-1],)

    def call(self, inputs, training=None):
        x, conditions = _unpack_inputs(inputs)

        # Merge input and conditions (conditional mode)
        if self.x_proj is not None:
            h = self.x_proj(x)
            if conditions is not None and self.c_proj is not None:
                hc = self.c_proj(conditions)
                if self.merge == "concat":
                    h = keras.ops.concatenate([h, hc], axis=-1)
                else:
                    h = h + hc
                h = self.merge_proj(self.input_act(h))
            h = self.input_act(h)
        else:
            h = x

        for block in self.blocks:
            h = block(h, training=training)

        return h
