from collections.abc import Sequence

import keras
from keras.saving import register_keras_serializable as serializable

from bayesflow.networks.summary_network import SummaryNetwork
from bayesflow.types import Tensor
from bayesflow.utils import logging
from bayesflow.utils.decorators import sanitize_input_shape

try:
    from mamba_ssm import Mamba
except ImportError:
    logging.error("Mamba class is not available. Please, install the mamba-ssm library via `pip install mamba-ssm`.")


@serializable("bayesflow.wrappers")
class MambaBlock(keras.Layer):
    """
    Wraps the original Mamba module from, with added functionality for bidirectional processing:
    https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/mamba_simple.py

    Copyright (c) 2023, Tri Dao, Albert Gu.
    """

    def __init__(
        self,
        state_dim: int,
        conv_dim: int,
        feature_dim: int = 16,
        expand: int = 2,
        bidirectional: bool = True,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        device: str = "cuda",
        **kwargs,
    ):
        """
        A Keras layer implementing a Mamba-based sequence processing block.

        This layer applies a Mamba model for sequence modeling, preceded by a
        convolutional projection and followed by layer normalization.

        Parameters
        ----------
        state_dim : int
            The dimension of the state space in the Mamba model.
        conv_dim : int
            The dimension of the convolutional layer used in Mamba.
        feature_dim : int, optional
            The feature dimension for input projection and Mamba processing (default is 16).
        expand : int, optional
            Expansion factor for Mamba's internal dimension (default is 1).
        dt_min : float, optional
            Minimum delta time for Mamba (default is 0.001).
        dt_max : float, optional
            Maximum delta time for Mamba (default is 0.1).
        device : str, optional
            The device to which the Mamba model is moved, typically "cuda" or "cpu" (default is "cuda").
        **kwargs : dict
            Additional keyword arguments passed to the `keras.layers.Layer` initializer.
        """

        super().__init__(**kwargs)

        if keras.backend.backend() != "torch":
            raise EnvironmentError("Mamba is only available using torch backend.")

        self.bidirectional = bidirectional

        self.mamba = Mamba(
            d_model=feature_dim, d_state=state_dim, d_conv=conv_dim, expand=expand, dt_min=dt_min, dt_max=dt_max
        ).to(device)

        self.input_projector = keras.layers.Conv1D(
            feature_dim,
            kernel_size=1,
            strides=1,
        )
        self.layer_norm = keras.layers.LayerNormalization()

    def call(self, x: Tensor, training: bool = False, **kwargs) -> Tensor:
        out_forward = self._call(x, training=training, **kwargs)
        if self.bidirectional:
            out_backward = self._call(keras.ops.flip(x, axis=1), training=training, **kwargs)
            return keras.ops.concatenate((out_forward, out_backward), axis=-1)
        return out_forward

    def _call(self, x: Tensor, training: bool = False, **kwargs) -> Tensor:
        x = self.input_projector(x)
        h = self.mamba(x)
        out = self.layer_norm(h + x, training=training, **kwargs)
        return out

    @sanitize_input_shape
    def build(self, input_shape):
        super().build(input_shape)
        self.call(keras.ops.zeros(input_shape))


@serializable("bayesflow.wrappers")
class MambaSSM(SummaryNetwork):
    """
    Wraps a sequence of Mamba modules using the simple Mamba module from:
    https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/mamba_simple.py

    Copyright (c) 2023, Tri Dao, Albert Gu.
    """

    def __init__(
        self,
        summary_dim: int = 16,
        feature_dims: Sequence[int] = (64, 64),
        state_dims: Sequence[int] = (64, 64),
        conv_dims: Sequence[int] = (64, 64),
        expand_dims: Sequence[int] = (2, 2),
        bidirectional: bool = True,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dropout: float = 0.05,
        device: str = "cuda",
        **kwargs,
    ):
        """
        A time-series summarization network using Mamba-based State Space Models (SSM). This model processes
        sequential input data using a sequence of Mamba SSM layers (determined by the length of the tuples),
        followed by optional pooling, dropout, and a dense layer for extracting summary statistics.

        Parameters
        ----------
        summary_dim : Sequence[int], optional
            The output dimensionality of the summary statistics layer (default is 16).
        feature_dims : Sequence[int], optional
            The feature dimension for each mamba block, default is (64, 64),
        state_dims : Sequence[int], optional
            The dimensionality of the internal state in each Mamba block, default is (64, 64)
        conv_dims : Sequence[int], optional
            The dimensionality of the convolutional layer in each Mamba block, default is (32, 32)
        expand_dims : Sequence[int], optional
            The expansion factors for the hidden state in each Mamba block, default is (2, 2)
        dt_min : float, optional
            Minimum dynamic state evolution over time (default is 0.001).
        dt_max : float, optional
            Maximum dynamic state evolution over time (default is 0.1).
        pooling : bool, optional
            Whether to apply global average pooling (default is True).
        dropout : int, float, or None, optional
            Dropout rate applied before the summary layer (default is 0.5).
        dropout: float, optional
            Dropout probability; dropout is applied to the pooled summary vector.
        device : str, optional
            The computing device. Currently, only "cuda" is supported (default is "cuda").
        **kwargs : dict
            Additional keyword arguments passed to the `SummaryNetwork` parent class.
        """

        super().__init__(**kwargs)

        if device != "cuda":
            raise NotImplementedError("MambaSSM only supports cuda as `device`.")

        self.mamba_blocks = []
        for feature_dim, state_dim, conv_dim, expand in zip(feature_dims, state_dims, conv_dims, expand_dims):
            mamba = MambaBlock(feature_dim, state_dim, conv_dim, expand, bidirectional, dt_min, dt_max, device)
            self.mamba_blocks.append(mamba)

        self.pooling_layer = keras.layers.GlobalAveragePooling1D()
        self.dropout = keras.layers.Dropout(dropout)
        self.summary_stats = keras.layers.Dense(summary_dim)

    def call(self, time_series: Tensor, training: bool = True, **kwargs) -> Tensor:
        """
        Apply a sequence of Mamba blocks, followed by pooling, dropout, and summary statistics.

        Parameters
        ----------
        time_series : Tensor
            Input tensor representing the time series data, typically of shape
            (batch_size, sequence_length, feature_dim).
        training : bool, optional
            Whether the model is in training mode (default is True). Affects behavior of
            layers like dropout.
        **kwargs : dict
            Additional keyword arguments (not used in this method).

        Returns
        -------
        Tensor
            Output tensor after applying Mamba blocks, pooling, dropout, and summary statistics.
        """

        summary = time_series
        for mamba_block in self.mamba_blocks:
            summary = mamba_block(summary, training=training)

        summary = self.pooling_layer(summary)
        summary = self.dropout(summary, training=training)
        summary = self.summary_stats(summary)

        return summary

    @sanitize_input_shape
    def build(self, input_shape):
        super().build(input_shape)
        self.call(keras.ops.zeros(input_shape))
