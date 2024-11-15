import keras

from bayesflow.types import Tensor


def jvp(fn: callable, primals: tuple[Tensor] | Tensor, tangents: tuple[Tensor] | Tensor):
    """Compute the dot product between the Jacobian of the given function at the point given by
    the input (primals) and vectors in tangents."""

    match keras.backend.backend():
        case "torch":
            import torch

            fn_output, _jvp = torch.autograd.functional.jvp(fn, primals, tangents)
        case "tensorflow":
            import tensorflow as tf

            with tf.autodiff.ForwardAccumulator(primals=primals, tangents=tangents) as acc:
                fn_output = fn(*primals)
            _jvp = acc.jvp(fn_output)
        case "jax":
            import jax

            fn_output, _jvp = jax.jvp(
                fn,
                primals,
                tangents,
            )
        case _:
            raise NotImplementedError(f"JVP not implemented for backend {keras.backend.backend()}")
    return fn_output, _jvp
