import keras
from keras.saving import register_keras_serializable as serializable

from bayesflow.types import Tensor
from bayesflow.utils import keras_kwargs, find_recurrent_net


@serializable(package="bayesflow.networks")
class SkipRecurrentNet(keras.Model):
    """
    Implements a Skip recurrent layer as described in [1], but allowing a more flexible
    recurrent backbone and a more flexible implementation.

    [1] Y. Zhang and L. Mikelsons, Solving Stochastic Inverse Problems with Stochastic BayesFlow,
    2023 IEEE/ASME International Conference on Advanced Intelligent Mechatronics (AIM),
    Seattle, WA, USA, 2023, pp. 966-972, doi: 10.1109/AIM46323.2023.10196190.

    TODO: Add proper docstring

    """

    def __init__(
        self,
        hidden_dim: int = 256,
        recurrent_type: str = "gru",
        bidirectional: bool = True,
        input_channels: int = 64,
        skip_steps: int = 4,
        dropout: float = 0.05,
        **kwargs,
    ):
        super().__init__(**keras_kwargs(kwargs))

        self.skip_conv = keras.layers.Conv1D(
            filters=input_channels * skip_steps,
            kernel_size=skip_steps,
            strides=skip_steps,
            padding="same",
        )

        recurrent_constructor = find_recurrent_net(recurrent_type)

        self.recurrent = recurrent_constructor(
            units=hidden_dim // 2 if bidirectional else hidden_dim,
            dropout=dropout,
        )
        self.skip_recurrent = recurrent_constructor(
            units=hidden_dim // 2 if bidirectional else hidden_dim,
            dropout=dropout,
        )
        if bidirectional:
            self.recurrent = keras.layers.Bidirectional(self.recurrent)
            self.skip_recurrent = keras.layers.Bidirectional(self.skip_recurrent)
        self.input_channels = input_channels

    def call(self, time_series: Tensor, training: bool = False, **kwargs) -> Tensor:
        direct_summary = self.recurrent(time_series, training=training)
        skip_summary = self.skip_recurrent(self.skip_conv(time_series), training=training)
        return keras.ops.concatenate((direct_summary, skip_summary), axis=-1)

    def build(self, input_shape):
        self.call(keras.ops.zeros(input_shape))
