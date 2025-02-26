import numpy as np

from bayesflow.utils.serialization import Serializable


class ElementwiseTransform(Serializable):
    """Base class on which other transforms are based"""

    def __init__(self):
        self.initialize_config()

    def __call__(self, data: np.ndarray, inverse: bool = False, **kwargs) -> np.ndarray:
        if inverse:
            return self.inverse(data, **kwargs)

        return self.forward(data, **kwargs)

    def forward(self, data: np.ndarray, **kwargs) -> np.ndarray:
        raise NotImplementedError

    def inverse(self, data: np.ndarray, **kwargs) -> np.ndarray:
        raise NotImplementedError
