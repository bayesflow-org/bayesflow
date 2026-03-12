import keras

from bayesflow.types import Tensor
from bayesflow.utils import find_recurrent_net, layer_kwargs
from bayesflow.utils.serialization import serializable, serialize, deserialize


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

        if bidirectional:
            # Manually implement bidirectional to avoid Keras serialization issues with Bidirectional
            forward_recurrent = recurrent_constructor(units=hidden_dim, dropout=dropout)
            backward_recurrent = recurrent_constructor(units=hidden_dim, dropout=dropout)
            self.recurrent_forward = forward_recurrent
            self.recurrent_backward = backward_recurrent

            # Same for skip recurrent
            forward_skip = recurrent_constructor(units=hidden_dim, dropout=dropout)
            backward_skip = recurrent_constructor(units=hidden_dim, dropout=dropout)
            self.skip_recurrent_forward = forward_skip
            self.skip_recurrent_backward = backward_skip

        else:
            self.recurrent = recurrent_constructor(units=hidden_dim, dropout=dropout)
            self.skip_recurrent = recurrent_constructor(units=hidden_dim, dropout=dropout)

        self.input_channels = input_channels
        self.hidden_dim = hidden_dim
        self.recurrent_type = recurrent_type
        self.bidirectional = bidirectional
        self.skip_steps = skip_steps
        self.dropout = dropout

    def call(self, time_series: Tensor, training: bool = False, **kwargs) -> Tensor:
        if self.bidirectional:
            # Forward pass for recurrent branch
            forward_direct = self.recurrent_forward(time_series, training=training)
            # Backward pass for recurrent branch using reversed time axis
            backward_direct = self.recurrent_backward(keras.ops.flip(time_series, axis=1), training=training)
            backward_direct = keras.ops.flip(backward_direct, axis=1)
            direct_summary = forward_direct + backward_direct

            # Forward pass for skip recurrent branch
            skip_conv_output = self.skip_conv(time_series)
            forward_skip = self.skip_recurrent_forward(skip_conv_output, training=training)

            # Backward pass for skip recurrent branch
            backward_skip = self.skip_recurrent_backward(keras.ops.flip(skip_conv_output, axis=1), training=training)
            backward_skip = keras.ops.flip(backward_skip, axis=1)
            skip_summary = forward_skip + backward_skip
        else:
            direct_summary = self.recurrent(time_series, training=training)
            skip_summary = self.skip_recurrent(self.skip_conv(time_series), training=training)

        return direct_summary + skip_summary

    def get_config(self):
        base_config = super().get_config()
        config = {
            "hidden_dim": self.hidden_dim,
            "recurrent_type": self.recurrent_type,
            "bidirectional": self.bidirectional,
            "input_channels": self.input_channels,
            "skip_steps": self.skip_steps,
            "dropout": self.dropout,
        }

        return base_config | serialize(config)

    @classmethod
    def from_config(cls, config, custom_objects=None):
        return cls(**deserialize(config, custom_objects=custom_objects))
