from numbers import Number

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

    def forward(self, data: any, **kwargs) -> Tensor:
        if self.original_type is None:
            self.original_type = type(data)
        return ops.convert_to_tensor(data)

    def inverse(self, data: Tensor, **kwargs) -> Tensor:
        if self.original_type is None:
            raise RuntimeError("Cannot call `inverse` before calling `forward` at least once.")

        if issubclass(self.original_type, Number):
            try:
                return self.original_type(data)
            except ValueError:
                pass
        return data
