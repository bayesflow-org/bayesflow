import keras
from keras import layers, Sequential
from keras.saving import register_keras_serializable as serializable

from bayesflow.types import Tensor

from .skip_recurrent import SkipRecurrentNet
from ..summary_network import SummaryNetwork


@serializable(package="bayesflow.networks")
class LSTNet(SummaryNetwork):
    """
    Implements a LSTNet Architecture as described in [1]

    [1] Y. Zhang and L. Mikelsons, Solving Stochastic Inverse Problems with Stochastic BayesFlow,
    2023 IEEE/ASME International Conference on Advanced Intelligent Mechatronics (AIM),
    Seattle, WA, USA, 2023, pp. 966-972, doi: 10.1109/AIM46323.2023.10196190.

    TODO: Add proper docstring

    """

    def __init__(
        self,
        summary_dim: int = 16,
        filters: int | list | tuple = 32,
        kernel_sizes: int | list | tuple = 3,
        strides: int | list | tuple = 1,
        activation: str = "mish",
        kernel_initializer: str = "glorot_uniform",
        groups: int = 8,
        recurrent_type: str | keras.Layer = "gru",
        recurrent_dim: int = 128,
        bidirectional: bool = True,
        dropout: float = 0.05,
        skip_steps: int = 4,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.summary_dim = summary_dim

        # Convolutional backbone -> can be extended with inception-like structure
        if not isinstance(filters, (list, tuple)):
            filters = (filters,)
        if not isinstance(kernel_sizes, (list, tuple)):
            kernel_sizes = (kernel_sizes,)
        if not isinstance(strides, (list, tuple)):
            strides = (strides,)
        self.conv = Sequential()
        for f, k, s in zip(filters, kernel_sizes, strides):
            self.conv.add(
                layers.Conv1D(
                    filters=f,
                    kernel_size=k,
                    strides=s,
                    activation=activation,
                    kernel_initializer=kernel_initializer,
                )
            )
            self.conv.add(layers.GroupNormalization(groups=groups))

        # Recurrent and feedforward backbones
        self.recurrent = SkipRecurrentNet(
            hidden_dim=recurrent_dim,
            recurrent_type=recurrent_type,
            bidirectional=bidirectional,
            input_channels=filters[-1],
            skip_steps=skip_steps,
            dropout=dropout,
        )
        self.output_projector = layers.Dense(summary_dim)

    def call(self, time_series: Tensor, **kwargs) -> Tensor:
        summary = self.conv(time_series, **kwargs)
        summary = self.recurrent(summary, **kwargs)
        summary = self.output_projector(summary)
        return summary

    def build(self, input_shape):
        super().build(input_shape)
        self.call(keras.ops.zeros(input_shape))
