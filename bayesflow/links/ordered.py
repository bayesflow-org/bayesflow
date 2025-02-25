import keras
from keras.saving import register_keras_serializable as serializable

from bayesflow.utils import keras_kwargs


@serializable(package="links.ordered")
class Ordered(keras.Layer):
    def __init__(self, axis: int, anchor_index: int, **kwargs):
        super().__init__(**keras_kwargs(kwargs))
        self.axis = axis
        self.anchor_index = anchor_index

        self.config = {"axis": axis, "anchor_index": anchor_index, **kwargs}

    def get_config(self):
        base_config = super().get_config()
        return base_config | self.config

    def build(self, input_shape):
        super().build(input_shape)

        assert self.anchor_index % input_shape[self.axis] != 0 and self.anchor_index != -1, (
            "anchor should not be first or last index."
        )
        self.group_indeces = dict(
            below=list(range(0, self.anchor_index)),
            above=list(range(self.anchor_index + 1, input_shape[self.axis])),
        )

    def call(self, inputs):
        # Divide in anchor, below and above
        below_inputs = keras.ops.take(inputs, self.group_indeces["below"], axis=self.axis)
        anchor_input = keras.ops.take(inputs, self.anchor_index, axis=self.axis)
        anchor_input = keras.ops.expand_dims(anchor_input, axis=self.axis)
        above_inputs = keras.ops.take(inputs, self.group_indeces["above"], axis=self.axis)

        # Apply softplus for positivity and cumulate to ensure ordered quantiles
        below = keras.activations.softplus(below_inputs)
        above = keras.activations.softplus(above_inputs)

        below = anchor_input - keras.ops.flip(keras.ops.cumsum(below, axis=self.axis), self.axis)
        above = anchor_input + keras.ops.cumsum(above, axis=self.axis)

        # Concatenate and reshape back
        x = keras.ops.concatenate([below, anchor_input, above], self.axis)
        return x

    def compute_output_shape(self, input_shape):
        return input_shape
