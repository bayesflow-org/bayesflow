import keras.ops as ops

from bayesflow.utils.serialization import serializable, serialize
from bayesflow.types import Tensor

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

    def forward(self, data: Tensor, **kwargs) -> Tensor:
        return ops.cast(data, self.to_dtype)

    def inverse(self, data: Tensor, **kwargs) -> Tensor:
        return ops.cast(data, self.from_dtype)
