import keras
import functools
from bayesflow.types.tensor import BackendTensor
from numpy import ndarray


def devices() -> list:
    """Returns a list of available GPU devices."""
    match keras.backend.backend():
        case "jax":
            import jax

            return jax.devices("gpu")
        case "tensorflow":
            import tensorflow as tf

            return tf.config.list_physical_devices("GPU")
        case "torch":
            import torch

            return [torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())]
        case "numpy":
            return []
        case _:
            raise NotImplementedError(f"Backend {keras.backend.backend()} not supported.")


@functools.lru_cache(maxsize=1)
def _backend_supports_float64() -> bool:
    backend = keras.backend.backend()
    if backend == "torch":
        import torch

        return not (torch.backends.mps.is_available() and not torch.cuda.is_available())
    if backend == "jax":
        import jax

        return jax.config.jax_enable_x64
    return True


def supported_dtype(dtype) -> str:
    dtype = keras.backend.standardize_dtype(dtype)
    if dtype == "float64" and not _backend_supports_float64():
        return "float32"
    return dtype


def move_tensor(tensor):
    if isinstance(tensor, BackendTensor):
        dtype = keras.ops.dtype(tensor)
    elif isinstance(tensor, ndarray):
        dtype = tensor.dtype
    else:
        return tensor
    tensor = keras.ops.cast(tensor, supported_dtype(dtype))
    return keras.ops.convert_to_tensor(tensor)
