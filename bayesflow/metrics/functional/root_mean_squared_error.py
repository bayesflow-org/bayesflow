import keras
from keras import ops

from bayesflow.types import Tensor


def root_mean_squared_error(x1: Tensor, x2: Tensor, normalize: bool = False, **kwargs) -> Tensor:
    """Computes the (normalized) root mean squared error between samples x1 and x2.

    :param x1: Tensor of shape (n, ...)

    :param x2: Tensor of shape (n, ...)

    :param normalize: Normalize the RMSE?

    :param kwargs: Currently ignored

    :return: Tensor of shape (n,)
        The RMSE between x1 and x2 over all remaining dimensions.
    """

    # cannot check first (batch) dimension since it will be unknown at compile time
    if keras.ops.shape(x1)[1:] != keras.ops.shape(x2)[1:]:
        raise ValueError(
            f"Expected x1 and x2 to have the same dimensions, "
            f"but got {keras.ops.shape(x1)[1:]} != {keras.ops.shape(x2)[1:]}."
        )

    # use flattened versions
    x1 = keras.ops.reshape(x1, (keras.ops.shape(x1)[0], -1))
    x2 = keras.ops.reshape(x2, (keras.ops.shape(x2)[0], -1))

    # TODO: how to normalize the RMSE?
    return ops.sqrt(ops.mean(ops.square(x1 - x2), axis=1))
