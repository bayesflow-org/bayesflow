import keras

from bayesflow.types import Tensor
from bayesflow.utils.serialization import serializable

from .skip_recurrent import SkipRecurrentNet
from ..summary_network import SummaryNetwork


@serializable("bayesflow.networks")
class TimeSeriesNetwork(SummaryNetwork):
    """
    (SN) Implements a LSTNet Architecture as described in [1]

    [1] Y. Zhang and L. Mikelsons, Solving Stochastic Inverse Problems with Stochastic BayesFlow,
    2023 IEEE/ASME International Conference on Advanced Intelligent Mechatronics (AIM),
    Seattle, WA, USA, 2023, pp. 966-972, doi: 10.1109/AIM46323.2023.10196190.
    """

    def __init__(
        self,
        summary_dim: int = 16,
        filters: int | list | tuple = 32,
        kernel_sizes: int | list | tuple = 3,
        strides: int | list | tuple = 1,
        activation: str = "mish",
        kernel_initializer: str = "glorot_uniform",
        groups: int = None,
        recurrent_type: str = "gru",
        recurrent_dim: int = 128,
        bidirectional: bool = True,
        dropout: float = 0.05,
        skip_steps: int = 4,
        **kwargs,
    ):
        """
        Initializes a hybrid convolutional-recurrent network for summarizing multivariate time series.

        This model combines a convolutional backbone with a recurrent module to efficiently process sequential data.

        The convolutional layers extract local features, which are then processed by a recurrent network for capturing
        long-range dependencies. The model supports various activation functions, kernel initializations, and
        bidirectional recurrence for enhanced representation learning.

        Parameters
        ----------
        summary_dim : int, optional
            Dimensionality of the final summary representation. Default is 16.
        filters : int, list, or tuple, optional
            Number of filters in each convolutional layer. If an integer is provided,
            all layers will have the same number of filters. Default is 32.
        kernel_sizes : int, list, or tuple, optional
            Size of the convolutional kernels. If an integer is provided, all layers
            will have the same kernel size. Default is 3.
        strides : int, list, or tuple, optional
            Stride length for convolutional layers. If an integer is provided, all layers
            will have the same stride. Default is 1.
        activation : str, optional
            Activation function applied in the convolutional layers. Default is "mish".
        kernel_initializer : str, optional
            Initialization strategy for convolutional kernels, such as "glorot_uniform".
            Default is "glorot_uniform".
        groups : int, optional
            Number of groups for group normalization applied after each convolutional layer.
            Default is None.
        recurrent_type : str, optional
            Type of recurrent layer used for sequence modeling, such as "gru" or "lstm".
            Default is "gru".
        recurrent_dim : int, optional
            Number of hidden units in the recurrent layer. Default is 128.
        bidirectional : bool, optional
            Whether to use a bidirectional recurrent network. Default is True.
        dropout : float, optional
            Dropout rate applied in the recurrent module. Default is 0.05.
        skip_steps : int, optional
            Number of steps to skip in the recurrent network for efficiency. Default is 4.
        **kwargs
            Additional keyword arguments passed to the parent class.
        """

        super().__init__(**kwargs)

        # Convolutional backbone -> can be extended with inception-like structure
        if not isinstance(filters, (list, tuple)):
            filters = (filters,)
        if not isinstance(kernel_sizes, (list, tuple)):
            kernel_sizes = (kernel_sizes,)
        if not isinstance(strides, (list, tuple)):
            strides = (strides,)

        conv_blocks = []
        for i, (f, k, s) in enumerate(zip(filters, kernel_sizes, strides)):
            conv_blocks.append(
                keras.layers.Conv1D(
                    filters=f,
                    kernel_size=k,
                    strides=s,
                    activation=activation,
                    kernel_initializer=kernel_initializer,
                    padding="same",
                    name=f"conv_{i}",
                )
            )
            if groups is not None:
                conv_blocks.append(keras.layers.GroupNormalization(groups=groups, name=f"group_norm_{i}"))

        self.conv = keras.Sequential(conv_blocks, name="conv")

        # Recurrent and feedforward backbones
        self.recurrent = SkipRecurrentNet(
            hidden_dim=recurrent_dim,
            recurrent_type=recurrent_type,
            bidirectional=bidirectional,
            input_channels=filters[-1],
            skip_steps=skip_steps,
            dropout=dropout,
            name="recurrent",
        )
        self.output_projector = keras.layers.Dense(summary_dim, name="output_projector")

        self.summary_dim = summary_dim

    def call(self, x: Tensor, training: bool = False, **kwargs) -> Tensor:
        """
        Performs the forward pass of the hybrid convolutional-recurrent network.

        This function applies a sequence of convolutional layers followed by a recurrent module to extract spatial
        and temporal features from the input tensor.

        The final output is projected into a lower-dimensional summary representation using a dense layer.

        Parameters
        ----------
        x : Tensor
            Input tensor representing the sequence data.
        training : bool, optional
            Whether the model is in training mode, affecting layers like dropout and
            batch normalization. Default is False.
        **kwargs
            Additional keyword arguments passed to the layers.

        Returns
        -------
        output: Tensor
            Transformed tensor representing the summarized feature representation
            of the input sequence.
        """
        x = self.conv(x, training=training)
        x = self.recurrent(x, training=training)
        x = self.output_projector(x)
        return x
