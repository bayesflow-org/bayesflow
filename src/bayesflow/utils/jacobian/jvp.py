from collections.abc import Callable
import keras

from bayesflow.types import Tensor


def jvp(
    f: Callable, x: Tensor | tuple[Tensor, ...], tangents: Tensor | tuple[Tensor, ...], return_output: bool = False
):
    """Compute the Jacobian-vector product of f at x with tangents."""
    if keras.ops.is_tensor(x):
        x = (x,)

    if keras.ops.is_tensor(tangents):
        tangents = (tangents,)

    match keras.backend.backend():
        case "torch":
            import torch

            fx, _jvp = torch.autograd.functional.jvp(f, x, tangents)
        case "tensorflow":
            import tensorflow as tf

            with tf.autodiff.ForwardAccumulator(primals=x, tangents=tangents) as acc:
                fx = f(*x)

            _jvp = acc.jvp(fx)
        case "jax":
            import jax

            fx, _jvp = jax.jvp(
                f,
                x,
                tangents,
            )
        case _:
            raise NotImplementedError(f"JVP not implemented for backend {keras.backend.backend()!r}")

    if return_output:
        return fx, _jvp

    return _jvp
