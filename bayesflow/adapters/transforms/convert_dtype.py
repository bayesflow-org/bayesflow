import numpy as np
from keras.tree import map_structure

from bayesflow.utils.serialization import serializable, serialize

from .elementwise_transform import ElementwiseTransform


@serializable("bayesflow.adapters")
class ConvertDType(ElementwiseTransform):
    """
    Default transform used to convert all floats from float64 to float32 to be in line with keras framework.

    Parameters
    ----------
    from_dtype : str
        Original dtype
    to_dtype : str
        Target dtype
    """

    def __init__(self, from_dtype: str, to_dtype: str):
        super().__init__()

        self.from_dtype = from_dtype
        self.to_dtype = to_dtype

    def get_config(self) -> dict:
        config = {
            "from_dtype": self.from_dtype,
            "to_dtype": self.to_dtype,
        }
        return serialize(config)

    def forward(self, data: np.ndarray | dict, **kwargs) -> np.ndarray | dict:
        return map_structure(lambda d: d.astype(self.to_dtype, copy=False), data)

    def inverse(self, data: np.ndarray | dict, **kwargs) -> np.ndarray | dict:
        return map_structure(lambda d: d.astype(self.from_dtype, copy=False), data)
