import keras
import numpy as np
from typing import TypeVar, Union

match keras.backend.backend():
    case "numpy":
        BackendTensor = np.ndarray
    case "jax":
        import jax

        BackendTensor = jax.Array
    case "tensorflow":
        import tensorflow as tf

        BackendTensor = tf.Tensor
    case "torch":
        import torch

        BackendTensor = torch.Tensor
    case other:
        raise NotImplementedError

Tensor = TypeVar("Tensor", bound=BackendTensor)

ArrayOrTensor = Union[np.ndarray, Tensor]
