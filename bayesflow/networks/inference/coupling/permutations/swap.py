import keras

from bayesflow.types import Shape
from bayesflow.utils.serialization import serializable

from .fixed_permutation import FixedPermutation


@serializable("bayesflow.networks")
class Swap(FixedPermutation):
    def build(self, xz_shape: Shape, **kwargs) -> None:
        shift = xz_shape[-1] // 2
        forward_indices = keras.ops.roll(keras.ops.arange(xz_shape[-1]), shift=shift)
        inverse_indices = keras.ops.argsort(forward_indices)

        self.forward_indices = self.add_variable(
            shape=(xz_shape[-1],),
            initializer=keras.initializers.Constant(forward_indices),
            trainable=False,
            dtype="int",
        )

        self.inverse_indices = self.add_variable(
            shape=(xz_shape[-1],),
            initializer=keras.initializers.Constant(inverse_indices),
            trainable=False,
            dtype="int",
        )
