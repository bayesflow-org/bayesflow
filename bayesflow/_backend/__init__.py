import importlib

import keras

try:
    backend = importlib.import_module(f"._{keras.backend.backend()}", package=__name__)

    jacfwd = backend.jacfwd
    jacrev = backend.jacrev
    jvp = backend.jvp
    jit = backend.jit
    grad = backend.grad
    value_and_grad = backend.value_and_grad
    vjp = backend.vjp

except ImportError as e:
    raise RuntimeError(f"Failed to import backend {keras.backend.backend()!r}") from e
except AttributeError as e:
    raise RuntimeError(f"Setup for backend {keras.backend.backend()!r} failed") from e
