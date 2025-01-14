from collections.abc import Callable
import keras

from bayesflow.types import Tensor


def vjp(f: Callable[[Tensor], Tensor], x: Tensor, return_output: bool = False):
    """Compute the vector-Jacobian product of f at x."""
    match keras.backend.backend():
        case "jax":
            import jax

            fx, _vjp_fn = jax.vjp(f, x)

            def vjp_fn(projector):
                return _vjp_fn(projector)[0]
        case "tensorflow":
            import tensorflow as tf

            with tf.GradientTape(persistent=True) as tape:
                tape.watch(x)
                fx = f(x)

            def vjp_fn(projector):
                return tape.gradient(fx, x, projector)
        case "torch":
            import torch

            x = keras.ops.copy(x)
            x.requires_grad_(True)

            with torch.enable_grad():
                fx = f(x)

            def vjp_fn(projector):
                return torch.autograd.grad(fx, x, projector, retain_graph=True)[0]
        case other:
            raise NotImplementedError(f"Cannot build a vjp function for backend '{other}'.")

    if return_output:
        return fx, vjp_fn

    return vjp_fn
