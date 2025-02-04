from collections.abc import Sequence

import keras
from keras.saving import register_keras_serializable as serializable
from mamba_ssm import Mamba

from bayesflow.types import Tensor
from ...networks.summary_network import SummaryNetwork


class MambaSSM(SummaryNetwork):
    def __init__(self, pooling: bool, dropout: int | float | None = 0.5, **kwargs):
        super().__init__(**kwargs)
        
        batch, length, dim = 64, 14, 1
        self.mamba = Mamba(d_model=dim, d_state=16, d_conv=4).to("cuda")
        self.pooling = keras.layers.GlobalAveragePooling1D().to("cuda")
        self.dropout = keras.layers.Dropout(dropout).to("cuda")
        self.summary_stats = keras.layers.Dense(8).to("cuda")
        
    def call(self, time_series, **kwargs):
        summary = self.mamba(time_series)
        summary = self.pooling(summary)
        summary = self.dropout(summary)
        summary = self.summary_stats(summary)
        return summary