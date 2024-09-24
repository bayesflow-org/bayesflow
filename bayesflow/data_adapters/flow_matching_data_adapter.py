from keras.saving import register_keras_serializable as serializable
import numpy as np
from typing import TypeVar

from bayesflow.utils import optimal_transport

from .data_adapter import DataAdapter


TRaw = TypeVar("TRaw")
TProcessed = dict[str, np.ndarray | tuple[np.ndarray, ...]]


@serializable(package="bayesflow.data_adapters")
class FlowMatchingDataAdapter(DataAdapter[TRaw, TProcessed]):
    """Wraps a data adapter, applying all further processing required for Optimal Transport Flow Matching.
    Useful to move these operations into a worker process, so as not to slow down training.
    """

    def __init__(self, inner: DataAdapter[TRaw, dict[str, np.ndarray]], key: str = "inference_variables", **kwargs):
        self.inner = inner
        self.key = key
        self.kwargs = kwargs

    def configure(self, raw_data: TRaw) -> TProcessed:
        processed_data = self.inner.configure(raw_data)

        x1 = processed_data[self.key]
        x0 = np.random.standard_normal(size=x1.shape).astype(x1.dtype)
        t = np.random.uniform(size=x1.shape[0]).astype(x1.dtype)

        expand_index = [slice(None)] + [None] * (x1.ndim - 1)
        t = t[tuple(expand_index)]

        x0, x1 = optimal_transport(x0, x1, **self.kwargs, numpy=True)

        x = t * x1 + (1 - t) * x0

        target_velocity = x1 - x0

        return processed_data | {self.key: (x0, x1, t, x, target_velocity)}

    def deconfigure(self, variables: TProcessed) -> TRaw:
        return self.inner.deconfigure(variables)
