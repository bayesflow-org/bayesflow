import numpy as np

from bayesflow.utils.serialization import serializable, deserialize


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

    def forward(self, data: np.ndarray, **kwargs) -> np.ndarray:
        raise NotImplementedError

    def inverse(self, data: np.ndarray, **kwargs) -> np.ndarray:
        raise NotImplementedError

    def log_det_jac(self, data: np.ndarray, inverse: bool = False, **kwargs) -> np.ndarray | None:
        return None
