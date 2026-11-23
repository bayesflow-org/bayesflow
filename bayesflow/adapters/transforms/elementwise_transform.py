import keras.ops as ops

from bayesflow.utils.serialization import serializable, deserialize
from bayesflow.types import Tensor
from typing import Union


@serializable("bayesflow.adapters")
class ElementwiseTransform:
    """Base class on which other transforms are based"""

    def __call__(self, data: Tensor, inverse: bool = False, **kwargs) -> Tensor:
        if inverse:
            return self.inverse(data, **kwargs)

        return self.forward(data, **kwargs)

    @classmethod
    def from_config(cls, config: dict, custom_objects=None):
        return cls(**deserialize(config, custom_objects=custom_objects))

    def get_config(self) -> dict:
        raise NotImplementedError

    def forward(self, data: Tensor, **kwargs) -> Tensor:
        raise NotImplementedError

    def inverse(self, data: Tensor, **kwargs) -> Tensor:
        raise NotImplementedError

    def log_det_jac(self, data: Tensor, inverse: bool = False, **kwargs) -> Union[Tensor, None]:
        return None

    @staticmethod
    def _sum_except_batch(data: Tensor) -> Tensor:
        b = ops.shape(data)[0]
        data = ops.reshape(data, (b, -1))
        return ops.sum(data, axis=1)
