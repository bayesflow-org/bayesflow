import numpy as np
from keras.saving import register_keras_serializable as serializable

from .elementwise_transform import ElementwiseTransform


@serializable(package="bayesflow.adapters")
class Take(ElementwiseTransform):
    """
    A transform to reduce the dimensionality of arrays output by the summary network
    Axis is a mandatory argument and will default  to the last axis.
    Example: adapter.take("x", np.arange(0,3), axis=-1)

    """

    def __init__(self,indices, axis=-1):
        super().__init__()
        self.indices = indices
        self.axis = axis 


    def forward(self, data):
        return np.take(data, self.indices, self.axis)

    def inverse(self, data):
        # not a true invertible function
        return data
