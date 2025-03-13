import keras


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
