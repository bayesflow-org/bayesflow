from typing import Literal

import keras
from keras import layers
from keras.saving import register_keras_serializable as serializable

from bayesflow.types import Tensor


@serializable(package="bayesflow.networks")
class ConfigurableHiddenBlock(keras.layers.Layer):
    def __init__(
        self,
        units: int = 256,
        activation: str = "mish",
        kernel_initializer: str = "he_normal",
        residual: bool = False,
        dropout: Literal[0, None] | float = 0.05,
        spectral_normalization: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.activation_fn = keras.activations.get(activation)
        self.residual = residual
        self.dense = layers.Dense(
            units=units,
            kernel_initializer=kernel_initializer,
        )
        if spectral_normalization:
            self.dense = layers.SpectralNormalization(self.dense)

        if dropout is not None and dropout > 0.0:
            self.dropout = layers.Dropout(float(dropout))
        else:
            self.dropout = None

        self.units = units

    def call(self, inputs: Tensor, training: bool = False, **kwargs) -> Tensor:
        x = self.dense(inputs, training=training)

        if self.dropout is not None:
            x = self.dropout(x, training=training)

        if self.residual:
            x = x + (inputs if self.projector is None else keras.ops.matmul(inputs, self.projector))

        return self.activation_fn(x)

    def build(self, input_shape):
        self.dense.build(input_shape)

        if input_shape[-1] != self.units and self.residual:
            self.projector = self.add_weight(
                shape=(input_shape[-1], self.units), initializer="glorot_uniform", trainable=True, name="projector"
            )
        else:
            self.projector = None

        if self.dropout is not None:
            self.dropout.build(self.dense.compute_output_shape(input_shape))

    def compute_output_shape(self, input_shape):
        return self.dense.compute_output_shape(input_shape)
