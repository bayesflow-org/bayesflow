from collections.abc import Sequence

import keras

from bayesflow.networks.summary_network import SummaryNetwork
from bayesflow.types import Tensor
from bayesflow.utils.serialization import serializable

from .mamba_block import MambaBlock


@serializable("bayesflow.wrappers")
class Mamba(SummaryNetwork):
    """
    (SN) Wraps a sequence of Mamba modules using the simple Mamba module from:
    https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/mamba_simple.py

    Copyright (c) 2023, Tri Dao, Albert Gu.

    Example usage in a BayesFlow workflow as a summary network:

    `summary_net = bayesflow.wrappers.Mamba(summary_dim=32)`
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
        **kwargs :
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

    def call(self, time_series: Tensor, training: bool = False, **kwargs) -> Tensor:
        """
        Apply a sequence of Mamba blocks, followed by pooling, dropout, and summary statistics.

        Parameters
        ----------
        time_series : Tensor
            Input tensor representing the time series data, typically of shape
            (batch_size, sequence_length, feature_dim).
        training : bool, optional
            Whether the model is in training mode (default is False). Affects the behavior of
            the inner dropout and norm layers.
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
