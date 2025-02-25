import keras

from bayesflow.utils import keras_kwargs


class PositiveSemiDefinite(keras.Layer):
    def __init__(self, **kwargs):
        super().__init__(**keras_kwargs(kwargs))

    def call(self, inputs):
        # add identity to avoid coliniarity of input columns
        inputs += keras.ops.identity(inputs.shape[-1])
        # multiply M * M^T to get symmetric matrix
        return keras.ops.einsum("...ij,...kj->...ik", inputs, inputs)

    def compute_output_shape(self, input_shape):
        return input_shape
