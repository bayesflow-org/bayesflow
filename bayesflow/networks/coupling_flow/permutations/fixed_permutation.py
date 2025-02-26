import keras

from bayesflow.types import Shape, Tensor
from ..invertible_layer import InvertibleLayer


class FixedPermutation(InvertibleLayer):
    def __init__(self, forward_indices=None, inverse_indices=None, **kwargs):
        super().__init__(**kwargs)
        self.initialize_config()

        self.forward_indices = forward_indices
        self.inverse_indices = inverse_indices

    def call(self, xz: Tensor, inverse: bool = False, **kwargs):
        if inverse:
            return self._inverse(xz)
        return self._forward(xz)

    def build(self, xz_shape: Shape, **kwargs) -> None:
        raise NotImplementedError

    def _forward(self, x: Tensor) -> (Tensor, Tensor):
        z = keras.ops.take(x, self.forward_indices, axis=-1)
        log_det = keras.ops.zeros(keras.ops.shape(x)[:-1])
        return z, log_det

    def _inverse(self, z: Tensor) -> (Tensor, Tensor):
        x = keras.ops.take(z, self.inverse_indices, axis=-1)
        log_det = keras.ops.zeros(keras.ops.shape(x)[:-1])
        return x, log_det
