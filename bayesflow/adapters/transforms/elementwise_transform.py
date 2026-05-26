import numpy as np
import keras.ops as ops

from bayesflow.utils.serialization import serializable, deserialize
from bayesflow.types import ArrayOrTensor, Tensor
from typing import Union


@serializable("bayesflow.adapters")
class ElementwiseTransform:
    """Base class on which other transforms are based"""

    def __call__(self, data: np.ndarray, inverse: bool = False, **kwargs) -> np.ndarray:
        if inverse:
            return self.inverse(data, **kwargs)

        return self.forward(data, **kwargs)

    @classmethod
    def from_config(cls, config: dict, custom_objects=None):
        return cls(**deserialize(config, custom_objects=custom_objects))

    def get_config(self) -> dict:
        raise NotImplementedError

    def forward(self, data: ArrayOrTensor, keras: bool = False, **kwargs) -> ArrayOrTensor:
        if keras:
            return self._forward_keras(data, **kwargs)
        else:
            return self._forward(data, **kwargs)

    def _forward(self, data: np.ndarray, **kwargs) -> np.ndarray:
        raise NotImplementedError

    def _forward_keras(self, data: Tensor, **kwargs) -> Tensor:
        raise NotImplementedError

    def inverse(self, data: ArrayOrTensor, keras: bool = False, **kwargs) -> ArrayOrTensor:
        if keras:
            return self._inverse_keras(data, **kwargs)
        else:
            return self._inverse(data, **kwargs)

    def _inverse(self, data: np.ndarray, **kwargs) -> np.ndarray:
        raise NotImplementedError

    def _inverse_keras(self, data: Tensor, **kwargs) -> Tensor:
        raise NotImplementedError

    def log_det_jac(
        self, data: ArrayOrTensor, inverse: bool = False, keras: bool = False, **kwargs
    ) -> Union[ArrayOrTensor, None]:
        if keras:
            return self._log_det_jac_keras(data, inverse=inverse, **kwargs)
        else:
            return self._log_det_jac(data, inverse=inverse, **kwargs)

    def _log_det_jac(self, data: np.ndarray, inverse: bool = False, **kwargs) -> Union[np.ndarray, None]:
        return None

    def _log_det_jac_keras(self, data: Tensor, inverse: bool = False, **kwargs) -> Union[Tensor, None]:
        return None

    @staticmethod
    def _sum_except_batch_keras(data: Tensor) -> Tensor:
        b = ops.shape(data)[0]
        data = ops.reshape(data, (b, -1))
        return ops.sum(data, axis=1)
