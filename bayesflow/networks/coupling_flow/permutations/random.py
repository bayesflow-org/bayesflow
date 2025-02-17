import keras
from keras.saving import register_keras_serializable as serializable

from bayesflow.types import Shape
from .fixed_permutation import FixedPermutation


@serializable(package="networks.coupling_flow")
class RandomPermutation(FixedPermutation):
    # noinspection PyMethodOverriding
    def build(self, xz_shape: Shape, **kwargs) -> None:
        forward_indices = keras.random.shuffle(keras.ops.arange(xz_shape[-1]))
        inverse_indices = keras.ops.argsort(forward_indices)

        self.forward_indices = self.add_weight(
            shape=(xz_shape[-1],),
            initializer=keras.initializers.Constant(forward_indices),
            trainable=False,
            dtype="int",
        )

        self.inverse_indices = self.add_weight(
            shape=(xz_shape[-1],),
            initializer=keras.initializers.Constant(inverse_indices),
            trainable=False,
            dtype="int",
        )
