import keras
from keras.saving import register_keras_serializable as serializable

from bayesflow.types import Tensor
from bayesflow.utils import keras_kwargs


@serializable(package="bayesflow.links")
class PositiveSemiDefinite(keras.Layer):
    """Activation function to link from any square matrix to a positive semidefinite matrix."""

    def __init__(self, **kwargs):
        super().__init__(**keras_kwargs(kwargs))

    def call(self, inputs: Tensor) -> Tensor:
        # multiply M * M^T to get symmetric matrix
        return keras.ops.einsum("...ij,...kj->...ik", inputs, inputs)

    def compute_output_shape(self, input_shape):
        return input_shape
