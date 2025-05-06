import keras

from bayesflow.types import Tensor
from bayesflow.utils import find_recurrent_net, layer_kwargs
from bayesflow.utils.decorators import sanitize_input_shape
from bayesflow.utils.serialization import serializable


@serializable("bayesflow.networks")
class SkipRecurrentNet(keras.Layer):
    """
    Implements a Skip recurrent layer as described in [1], allowing a more flexible recurrent backbone
    and a more efficient implementation.

    [1] Y. Zhang and L. Mikelsons, Solving Stochastic Inverse Problems with Stochastic BayesFlow,
    2023 IEEE/ASME International Conference on Advanced Intelligent Mechatronics (AIM),
    Seattle, WA, USA, 2023, pp. 966-972, doi: 10.1109/AIM46323.2023.10196190.
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
        """
        Creates a skip recurrent neural network layer that extends a traditional recurrent backbone with
        skip connections implemented via convolution and an additional recurrent path. This allows
        more efficient modeling of long-term dependencies by combining local and non-local temporal
        features.

        Parameters
        ----------
        hidden_dim : int, optional
            Dimensionality of the hidden state in the recurrent layers. Default is 256.
        recurrent_type : str, optional
            Type of recurrent unit to use. Should correspond to a supported type in `find_recurrent_net`,
            such as "gru" or "lstm". Default is "gru".
        bidirectional : bool, optional
            If True, uses bidirectional wrappers for both recurrent and skip recurrent layers. Default is True.
        input_channels : int, optional
            Number of input channels for the 1D convolution used in skip connections. Default is 64.
        skip_steps : int, optional
            Step size and kernel size used in the skip convolution. Determines how many steps are skipped.
            Also determines the multiplier for the number of filters. Default is 4.
        dropout : float, optional
            Dropout rate applied within the recurrent layers. Default is 0.05.
        **kwargs
            Additional keyword arguments passed to the parent class constructor.
        """
        super().__init__(**layer_kwargs(kwargs))

        self.skip_conv = keras.layers.Conv1D(
            filters=input_channels * skip_steps,
            kernel_size=skip_steps,
            strides=skip_steps,
            padding="same",
            name="skip_conv",
        )

        recurrent_constructor = find_recurrent_net(recurrent_type)

        recurrent = recurrent_constructor(
            units=hidden_dim // 2 if bidirectional else hidden_dim,
            dropout=dropout,
        )
        recurrent.name = "recurrent"

        skip_recurrent = recurrent_constructor(
            units=hidden_dim // 2 if bidirectional else hidden_dim,
            dropout=dropout,
        )
        skip_recurrent.name = "skip_recurrent"

        if bidirectional:
            recurrent = keras.layers.Bidirectional(recurrent, name="bidirectional_recurrent")
            skip_recurrent = keras.layers.Bidirectional(skip_recurrent, name="bidirectional_skip_recurrent")

        self.recurrent = recurrent
        self.skip_recurrent = skip_recurrent
        self.input_channels = input_channels

    def call(self, time_series: Tensor, training: bool = False, **kwargs) -> Tensor:
        direct_summary = self.recurrent(time_series, training=training)
        skip_summary = self.skip_recurrent(self.skip_conv(time_series), training=training)
        return keras.ops.concatenate((direct_summary, skip_summary), axis=-1)

    @sanitize_input_shape
    def build(self, input_shape):
        super().build(input_shape)
