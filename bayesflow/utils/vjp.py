from collections.abc import Callable
import keras
from functools import partial

from bayesflow.types import Tensor


def vjp(fn: Callable, *primals: Tensor) -> (any, Callable[[Tensor], tuple[Tensor, ...]]):
    """
    Backend-agnostic version of the vector-Jacobian product (vjp).
    Computes the vector-Jacobian product of the given function at the point given by the input (primals).

    :param fn: The function to differentiate.
        Signature and return value must be compatible with the vjp method of the backend in use.

    :param primals: Input tensors to `fn`.

    :return: The output of `fn(*primals)` and a vjp function.
        The vjp function takes a single tensor argument, and returns the vector-Jacobian product of this argument with
        `fn` as evaluated at `primals`.
    """
    match keras.backend.backend():
        case "jax":
            import jax

            fx, vjp_fn = jax.vjp(fn, *primals)
        case "torch":
            import torch

            fx, vjp_fn = torch.func.vjp(fn, *primals)
        case "tensorflow":
            import tensorflow as tf

            with tf.GradientTape(persistent=True) as tape:
                for p in primals:
                    tape.watch(p)
                fx = fn(*primals)
                vjp_fn = partial(tape.gradient, fx, primals)
        case _:
            raise NotImplementedError(f"VJP not implemented for backend {keras.backend.backend()}")

    return fx, vjp_fn
