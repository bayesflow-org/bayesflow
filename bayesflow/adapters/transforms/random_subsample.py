import numpy as np
from bayesflow.utils.serialization import serializable
from .elementwise_transform import ElementwiseTransform


@serializable(package="bayesflow.adapters")
class RandomSubsample(ElementwiseTransform):
    """
    A transform that takes a random subsample of the data within an axis.

    Example: adapter.random_subsample("x", sample_size = 3, axis = -1)

    """

    def __init__(
        self,
        sample_size: int | float,
        axis: int = -1,
    ):
        super().__init__()
        if isinstance(sample_size, float):
            if sample_size <= 0 or sample_size >= 1:
                ValueError("Sample size as a percentage must be a float between 0 and 1 exclusive. ")
        self.sample_size = sample_size
        self.axis = axis

    def forward(self, data: np.ndarray, **kwargs) -> np.ndarray:
        axis = self.axis
        max_sample_size = data.shape[axis]

        if isinstance(self.sample_size, int):
            sample_size = self.sample_size
        else:
            sample_size = np.round(self.sample_size * max_sample_size)

        # random sample without replacement
        sample_indices = np.random.permutation(max_sample_size)[0 : sample_size - 1]

        return np.take(data, sample_indices, axis)

    def inverse(self, data: np.ndarray, **kwargs) -> np.ndarray:
        # non invertible transform
        return data
