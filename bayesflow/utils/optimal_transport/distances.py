import keras

from bayesflow.types import Tensor


def euclidean(x1: Tensor, x2: Tensor) -> Tensor:
    # TODO: rename and move this function
    result = x1[:, None] - x2[None, :]
    shape = list(keras.ops.shape(result))
    shape[2:] = [-1]
    result = keras.ops.reshape(result, shape)
    result = keras.ops.norm(result, ord=2, axis=-1)
    return result


def cosine_distance(x1: Tensor, x2: Tensor, eps: float = 1e-8) -> Tensor:
    """
    Pairwise cosine distance:
        d(x, y) = 1 - <x, y> / (||x|| ||y||)

    x1: Tensor of shape (n, ...)
    x2: Tensor of shape (m, ...)
    returns: Tensor of shape (n, m)
    """
    # Flatten trailing dimensions
    x1_flat = keras.ops.reshape(x1, (keras.ops.shape(x1)[0], -1))
    x2_flat = keras.ops.reshape(x2, (keras.ops.shape(x2)[0], -1))

    # L2 norms
    x1_norm = keras.ops.norm(x1_flat, ord=2, axis=1, keepdims=True)  # (n, 1)
    x2_norm = keras.ops.norm(x2_flat, ord=2, axis=1, keepdims=True)  # (m, 1)

    # Dot products (n, m)
    dot = keras.ops.matmul(x1_flat, keras.ops.transpose(x2_flat))

    # Cosine similarity and distance
    denom = keras.ops.maximum(x1_norm * keras.ops.transpose(x2_norm), eps)
    cos_sim = dot / denom
    return 1.0 - cos_sim
