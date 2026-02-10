import keras
from typing import TypeVar

match keras.backend.backend():
    case "numpy":
        import numpy as np

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
