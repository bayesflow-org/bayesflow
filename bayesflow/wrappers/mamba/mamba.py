import keras

# from keras.saving import register_keras_serializable as serializable
try:
    from mamba_ssm import Mamba
except ImportError:
    print("Mamba Wrapper is not available")

from ...networks.summary_network import SummaryNetwork


# @serializable(package="bayesflow.wrappers")
class MambaSSM(SummaryNetwork):
    def __init__(
        self,
        feature_dim: int,
        summary_dim: int = 8,
        mamba_blocks: int = 2,
        state_dim: int = 16,
        conv_dim: int = 4,
        expand: int = 2,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        pooling: bool = True,
        dropout: int | float | None = 0.5,
        device: str = "cuda",
        **kwargs,
    ):
        """
        A time-series summarization network using Mamba-based State Space Models (SSM).
        This model processes sequential input data using the Mamba SSM layer, followed by
        optional pooling, dropout, and a dense layer for extracting summary statistics.

        Mamba2 support currently unabailble due to stability issues

        Parameters
        ----------
        feature_dim : int
            The dimensionality of the Mamba SSM model.
        summary_dim : int, optional
            The output dimensionality of the summary statistics layer (default is 8).
        mamba_blocks : int, optional
            The number of sequential mamba blocks to use (default is 2).
        state_dim : int, optional
            The dimensionality of the internal state representation (default is 16).
        conv_dim : int, optional
            The dimensionality of the convolutional layer in Mamba (default is 4).
        expand : int, optional
            The expansion factor for the hidden state in Mamba (default is 2).
        dt_min : float, optional
            Minimum dynamic state evolution over time (default is 0.001).
        dt_max : float, optional
            Maximum dynmaic state evolution over time (default is 0.1).
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

        self.mamba_blocks = [
            Mamba(
                d_model=feature_dim, d_state=state_dim, d_conv=conv_dim, expand=expand, dt_min=dt_min, dt_max=dt_max
            ).to(device)
            for _ in range(mamba_blocks)
        ]

        self.layernorm = keras.layers.LayerNormalization(axis=-1)

        self.pooling = pooling
        if pooling:
            self.pooling = keras.layers.GlobalAveragePooling1D()
        self.dropout = keras.layers.Dropout(dropout)
        self.summary_stats = keras.layers.Dense(summary_dim)

    def call(self, time_series, **kwargs):
        summary = time_series
        for mamba_block in self.mamba_blocks:
            summary = self.layernorm(summary)
            summary = mamba_block(summary, **kwargs)
            summary = summary + 0.1 * time_series

        if self.pooling:
            summary = self.pooling(summary)
        summary = self.dropout(summary, **kwargs)
        summary = self.summary_stats(summary)
        return summary
