from numbers import Number

import numpy as np
import keras.ops as ops

from bayesflow.utils.serialization import serializable, serialize
from bayesflow.types import Tensor

from .elementwise_transform import ElementwiseTransform


@serializable("bayesflow.adapters")
class ToArray(ElementwiseTransform):
    """
    Checks provided data for any non-arrays and converts them to numpy arrays.

    This ensures all data is in a format suitable for training.

    Examples
    --------
    >>> ta = bf.adapters.transforms.ToArray()
    >>> a = [1, 2, 3, 4]
    >>> ta.forward(a)
        array([1, 2, 3, 4])
    >>> b = [[1, 2], [3, 4]]
    >>> ta.forward(b)
        array([[1, 2],
            [3, 4]])
    """

    def __init__(self, original_type: type = None):
        super().__init__()
        self.original_type = original_type

    def get_config(self) -> dict:
        return serialize({"original_type": self.original_type})

    def _forward(self, data: any, **kwargs) -> np.ndarray:
        if self.original_type is None:
            self.original_type = type(data)
        return np.asarray(data)

    def _forward_keras(self, data: any, **kwargs) -> Tensor:
        if self.original_type is None:
            self.original_type = type(data)
        return ops.convert_to_tensor(data)

    def _inverse(self, data: np.ndarray, **kwargs) -> any:
        if self.original_type is None:
            raise RuntimeError("Cannot call `inverse` before calling `forward` at least once.")

        if issubclass(self.original_type, Number):
            try:
                return self.original_type(data.item())
            except ValueError:
                pass

        return data

    def _inverse_keras(self, data: Tensor, **kwargs) -> Tensor:
        return data
