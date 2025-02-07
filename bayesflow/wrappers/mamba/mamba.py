from collections.abc import Sequence

import keras
from keras.saving import register_keras_serializable as serializable
try:
    from mamba_ssm import Mamba
except ImportError:
    print("Mamba Wrapper is not available")

from bayesflow.types import Tensor
from ...networks.summary_network import SummaryNetwork

# TODO: Also add Mamba2 model

# @serializable(package="bayesflow.wrappers")
class MambaSSM(SummaryNetwork):
    def __init__(
        self,
        ssm_dim: int,
        state_dim: int = 16,
        conv_dim: int = 4,
        expand: int = 2,
        summary_dim: int = 8,
        pooling: bool = True,
        dropout: int | float | None = 0.5,
        device: str = "cuda", 
        **kwargs
    ):
        """
        A time-series summarization network using Mamba-based State Space Models (SSM).
        This model processes sequential input data using the Mamba SSM layer, followed by 
        optional pooling, dropout, and a dense layer for extracting summary statistics.

        Parameters
        ----------
        ssm_dim : int
            The dimensionality of the Mamba SSM model.
        state_dim : int, optional
            The dimensionality of the internal state representation (default is 16).
        conv_dim : int, optional
            The dimensionality of the convolutional layer in Mamba (default is 4).
        expand : int, optional
            The expansion factor for the hidden state in Mamba (default is 2).
        summary_dim : int, optional
            The output dimensionality of the summary statistics layer (default is 8).
        pooling : bool, optional
            Whether to apply global average pooling (default is True).
        dropout : int, float, or None, optional
            Dropout rate applied before the summary layer (default is 0.5).
        device : str, optional
            The computing device. Currently, only "cuda" is supported (default is "cuda").
        **kwargs : dict
            Additional keyword arguments passed to the `SummaryNetwork` parent class.
        """
        
        super().__init__(**kwargs)
        if device != "cuda":
            raise NotImplementedError("MambaSSM currently only supports cuda")
        
        self.mamba = Mamba(d_model=ssm_dim, d_state=state_dim, d_conv=conv_dim, expand=expand)
        self.pooling = pooling
        if pooling:
            self.pooling = keras.layers.GlobalAveragePooling1D()
        self.dropout = keras.layers.Dropout(dropout)
        self.summary_stats = keras.layers.Dense(summary_dim)
        
    def call(self, time_series, **kwargs):
        summary = self.mamba(time_series, **kwargs)
        if self.pooling:
            summary = self.pooling(summary)
        summary = self.dropout(summary, **kwargs)
        summary = self.summary_stats(summary)
        return summary