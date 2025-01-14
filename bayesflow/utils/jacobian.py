from collections.abc import Callable
import keras
from keras import ops
from bayesflow.types import Tensor

from functools import partial, wraps


def compute_jacobian(
    x_in: Tensor,
    fn: Callable,
    *func_args: any,
    grad_type: str = "backward",
    **func_kwargs: any,
) -> tuple[Tensor, Tensor]:
    """Computes the Jacobian of a function with respect to its input.

    :param x_in: The input tensor to compute the jacobian at.
        Shape: (batch_size, in_dim).
    :param fn: The function to compute the jacobian of, which transforms
        `x` to `fn(x)` of shape (batch_size, out_dim).
    :param func_args: The positional arguments to pass to the function.
        func_args are batched over the first dimension.
    :param grad_type: The type of gradient to use. Either 'backward' or
        'forward'.
    :param func_kwargs: The keyword arguments to pass to the function.
        func_kwargs are not batched.
    :return: The output of the function `fn(x)` and the jacobian
        of the function with respect to its input `x` of shape
        (batch_size, out_dim, in_dim)."""

    def batch_wrap(fn: Callable) -> Callable:
        """Add a batch dimension to each tensor argument.

        :param fn:
        :return: wrapped function"""

        def deep_unsqueeze(arg):
            if ops.is_tensor(arg):
                return arg[None, ...]
            elif isinstance(arg, dict):
                return {key: deep_unsqueeze(value) for key, value in arg.items()}
            elif isinstance(arg, (list, tuple)):
                return [deep_unsqueeze(value) for value in arg]
            raise ValueError(f"Argument cannot be batched: {arg}")

        @wraps(fn)
        def wrapper(*args, **kwargs):
            args = deep_unsqueeze(args)
            return fn(*args, **kwargs)[0]

        return wrapper

    def double_output(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            out = fn(*args, **kwargs)
            return out, out

        return wrapper

    match keras.backend.backend():
        case "torch":
            import torch
            from torch.func import jacrev, jacfwd, vmap

            jacfn = jacrev if grad_type == "backward" else jacfwd
            with torch.inference_mode(False):
                with torch.no_grad():
                    fn_kwargs_prefilled = partial(fn, **func_kwargs)
                    fn_batch_expanded = batch_wrap(fn_kwargs_prefilled)
                    fn_return_val = double_output(fn_batch_expanded)
                    fn_jac_batched = vmap(jacfn(fn_return_val, has_aux=True))
                    jac, x_out = fn_jac_batched(x_in, *func_args)
        case "jax":
            from jax import jacrev, jacfwd, vmap

            jacfn = jacrev if grad_type == "backward" else jacfwd
            fn_kwargs_prefilled = partial(fn, **func_kwargs)
            fn_batch_expanded = batch_wrap(fn_kwargs_prefilled)
            fn_return_val = double_output(fn_batch_expanded)
            fn_jac_batched = vmap(jacfn(fn_return_val, has_aux=True))
            jac, x_out = fn_jac_batched(x_in, *func_args)
        case "tensorflow":
            if grad_type == "forward":
                raise NotImplementedError("For TensorFlow, only backward mode Jacobian computation is available.")
            import tensorflow as tf

            with tf.GradientTape() as tape:
                tape.watch(x_in)
                x_out = fn(x_in, *func_args, **func_kwargs)
            jac = tape.batch_jacobian(x_out, x_in)

        case _:
            raise NotImplementedError(f"compute_jacobian not implemented for {keras.backend.backend()}.")
    return x_out, jac


def log_jacobian_determinant(
    x_in: Tensor,
    fn: Callable,
    *func_args: any,
    grad_type: str = "backward",
    **func_kwargs: any,
) -> tuple[Tensor, Tensor]:
    """Computes the log Jacobian determinant of a function
    with respect to its input.

    :param x_in: The input tensor to compute the jacobian at.
        Shape: (batch_size, in_dim).
    :param fn: The function to compute the jacobian of, which transforms
        `x` to `fn(x)` of shape (batch_size, out_dim).
    :param func_args: The positional arguments to pass to the function.
        func_args are batched over the first dimension.
    :param grad_type: The type of gradient to use. Either 'backward' or
        'forward'.
    :param func_kwargs: The keyword arguments to pass to the function.
        func_kwargs are not batched.
    :return: The output of the function `fn(x)` and the log jacobian determinant
        of the function with respect to its input `x` of shape
        (batch_size, out_dim, in_dim)."""

    x_out, jac = compute_jacobian(x_in, fn, *func_args, grad_type=grad_type, **func_kwargs)
    jac = ops.reshape(
        jac, (ops.shape(x_in)[0], ops.prod(list(ops.shape(x_out)[1:])), ops.prod(list(ops.shape(x_in)[1:])))
    )
    log_det = ops.slogdet(jac)[1]

    return x_out, log_det
