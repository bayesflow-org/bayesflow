import keras

from bayesflow.types import Shape
from bayesflow.utils.serialization import serializable

from .fixed_permutation import FixedPermutation


@serializable("bayesflow.networks")
class RandomPermutation(FixedPermutation):
    """Random permutation of the target dimensions.

    The permutation is drawn once when the layer is built and then stored as a
    non-trainable weight.
    """

    # noinspection PyMethodOverriding
    def build(self, xz_shape: Shape, **kwargs) -> None:
        forward_indices = keras.random.shuffle(keras.ops.arange(xz_shape[-1]))
        inverse_indices = keras.ops.argsort(forward_indices)

        self.forward_indices = self.add_weight(
            shape=(xz_shape[-1],),
            # Best practice: https://github.com/keras-team/keras/pull/20457#discussion_r1832081248
            initializer=keras.initializers.get(forward_indices),
            trainable=False,
            dtype="int",
        )

        self.inverse_indices = self.add_weight(
            shape=(xz_shape[-1],),
            # Best practice: https://github.com/keras-team/keras/pull/20457#discussion_r1832081248
            initializer=keras.initializers.get(inverse_indices),
            trainable=False,
            dtype="int",
        )
