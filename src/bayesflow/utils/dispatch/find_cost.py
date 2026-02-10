from functools import singledispatch
import keras
import numpy as np

from bayesflow.types import Tensor
from bayesflow.types.tensor import BackendTensor


@singledispatch
def find_cost(arg, *args, **kwargs):
    raise TypeError(f"Cannot infer cost matrix from {arg!r}.")


@find_cost.register
def _(name: str, x1: Tensor, x2: Tensor, numpy: bool = False, **kwargs):
    match name.lower():
        case "euclidean":
            if numpy:
                cost = x1[:, None] - x2[None, :]
                cost = np.reshape(cost, (cost.shape[0], cost.shape[1], -1))
                cost = np.linalg.norm(cost, ord=2, axis=-1)
            else:
                cost = x1[:, None] - x2[None, :]
                cost = keras.ops.reshape(cost, (keras.ops.shape(cost)[0], keras.ops.shape(cost)[1], -1))
                cost = keras.ops.norm(cost, ord=2, axis=-1)
        case other:
            raise ValueError(f"Unsupported cost name: '{other}'.")

    return cost


@find_cost.register(BackendTensor)
def _(cost: Tensor, *args, **kwargs):
    return cost


@find_cost.register(np.ndarray)
def _(cost: np.ndarray, *args, **kwargs):
    return cost
