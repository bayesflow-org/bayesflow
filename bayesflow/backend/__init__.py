import keras
import importlib

try:
    backend = importlib.import_module(f"._{keras.backend.backend()}", package=__name__)

    grad = backend.grad
    value_and_grad = backend.value_and_grad

except ImportError as e:
    raise RuntimeError(f"Failed to import backend {keras.backend.backend()!r}") from e
except AttributeError as e:
    raise RuntimeError(f"Setup for backend {keras.backend.backend()!r} failed:") from e
