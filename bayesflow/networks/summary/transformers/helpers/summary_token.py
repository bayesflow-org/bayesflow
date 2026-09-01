import keras

from bayesflow.types import Tensor
from bayesflow.utils.serialization import serializable, serialize


@serializable("bayesflow.networks")
class SummaryToken(keras.Layer):
    """Learned sequence token used as a many-to-one transformer summary."""

    def __init__(self, kernel_initializer: str = "orthogonal", **kwargs):
        super().__init__(**kwargs)
        self.kernel_initializer = kernel_initializer

    def build(self, input_shape):
        if self.built:
            return

        self.summary_token = self.add_weight(
            shape=(1, 1, input_shape[-1]),
            initializer=self.kernel_initializer,
            trainable=True,
        )

    def call(self, x: Tensor) -> Tensor:
        batch_size = keras.ops.shape(x)[0]
        token = keras.ops.tile(self.summary_token, [batch_size, 1, 1])
        return keras.ops.concatenate([x, token], axis=1)

    @staticmethod
    def update_mask(mask: Tensor | None) -> Tensor | None:
        if mask is None:
            return None

        mask = keras.ops.cast(mask, "bool")
        batch_size = keras.ops.shape(mask)[0]

        token_mask = keras.ops.ones((batch_size, 1, 1), dtype="bool")
        return keras.ops.concatenate([mask, token_mask], axis=-1)

    @staticmethod
    def take(x: Tensor) -> Tensor:
        return x[:, -1, :]

    def get_config(self) -> dict:
        base_config = super().get_config()
        return base_config | serialize({"kernel_initializer": self.kernel_initializer})
