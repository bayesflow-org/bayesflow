from typing import Literal, Callable, Sequence

import keras

from bayesflow.types import Tensor
from bayesflow.utils import layer_kwargs
from bayesflow.utils.serialization import deserialize, serializable, serialize

from ...helpers import DenseBlock


@serializable("bayesflow.networks")
class MLP(keras.Layer):
    """Multi-layer perceptron built from :class:`DenseBlock` layers.

    Accepts a single tensor input.  When used inside a coupling layer the
    caller (e.g. :class:`SingleCoupling`) concatenates any non-time
    conditions onto the input before passing it here.

    Parameters
    ----------
    widths : Sequence[int], optional
        Number of hidden units per layer. Default is ``(256, 256)``.
    activation : str or callable, optional
        Activation function for hidden layers. Default is ``"mish"``.
    kernel_initializer : str or keras.Initializer, optional
        Weight initialization strategy. Default is ``"he_normal"``.
    residual : bool, optional
        Whether to use residual (skip) connections. Default is ``True``.
    dropout : float or None, optional
        Dropout rate for regularization. Default is ``0.05``.
    norm : ``"batch"``, ``"layer"``, ``"rms"``, keras.Layer, or None, optional
        Normalization applied after each hidden layer. Default is ``None``.
    spectral_normalization : bool, optional
        Apply spectral normalization to Dense layers. Default is ``False``.
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
        norm: Literal["batch", "layer", "rms"] | keras.Layer = None,
        spectral_normalization: bool = False,
        **kwargs,
    ):
        super().__init__(**layer_kwargs(kwargs))

        if len(widths) == 0:
            raise ValueError("MLP requires at least one hidden width.")

        self.widths = widths
        self.activation = activation
        self.kernel_initializer = kernel_initializer
        self.residual = residual
        self.dropout = dropout
        self.norm = norm
        self.spectral_normalization = spectral_normalization

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

    def call(self, x: Tensor, training: bool = None) -> Tensor:
        h = x
        for block in self.blocks:
            h = block(h, training=training)
        return h

    def build(self, input_shape):
        if self.built:
            return

        h_shape = input_shape

        for block in self.blocks:
            block.build(h_shape)
            h_shape = block.compute_output_shape(h_shape)

    def compute_output_shape(self, input_shape):
        h_shape = input_shape
        for block in self.blocks:
            h_shape = block.compute_output_shape(h_shape)
        return h_shape

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
        }
        return base_config | serialize(config)
