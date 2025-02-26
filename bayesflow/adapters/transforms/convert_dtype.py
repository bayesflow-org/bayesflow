import numpy as np

from .elementwise_transform import ElementwiseTransform


class ConvertDType(ElementwiseTransform):
    """
    Default transform used to convert all floats from float64 to float32 to be in line with keras framework.
    """

    def __init__(self, from_dtype: str, to_dtype: str):
        super().__init__()
        self.initialize_config()

        self.from_dtype = from_dtype
        self.to_dtype = to_dtype

    def forward(self, data: np.ndarray, **kwargs) -> np.ndarray:
        return data.astype(self.to_dtype)

    def inverse(self, data: np.ndarray, **kwargs) -> np.ndarray:
        return data.astype(self.from_dtype)
