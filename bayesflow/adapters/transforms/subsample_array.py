import numpy as np
from keras.saving import register_keras_serializable as serializable

from .elementwise_transform import ElementwiseTransform


@serializable(package="bayesflow.adapters")
class SubsampleArray(ElementwiseTransform):
    """
    A transform that takes a random subsample of the data within an axis.

    Example: adapter.subsample("x", sample_size = 3, axis = -1)

    """

    def __init__(self):
        super().__init__()

    def forward(self, data: np.ndarray, sample_size: int, axis=-1):
        max_sample_size = data.shape[axis]

        sample_indices = np.random.permutation(max_sample_size)[
            0 : sample_size - 1
        ]  # random sample without replacement

        return np.take(data, sample_indices, axis)

    def inverse(self, data, **kwargs):
        # non invertible transform
        return data
