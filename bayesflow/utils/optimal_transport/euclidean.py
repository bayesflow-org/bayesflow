import keras


def euclidean(x1, x2):
    # TODO: rename and move this function
    result = x1[:, None] - x2[None, :]
    shape = list(keras.ops.shape(result))
    shape[2:] = [-1]
    result = keras.ops.reshape(result, shape)
    result = keras.ops.norm(result, ord=2, axis=-1)
    return result
