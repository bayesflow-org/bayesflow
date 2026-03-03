import tensorflow as tf

from functools import wraps


def grad(fn, argnums=0, has_aux=False):
    grad_fn = value_and_grad(fn, argnums=argnums, has_aux=has_aux)

    @wraps(fn)
    def wrapper(*args, **kwargs):
        y, dydx = grad_fn(*args, **kwargs)
        return dydx

    return wrapper


def value_and_grad(fn, argnums=0, has_aux=False):
    if isinstance(argnums, int):
        argnums = [argnums]

    @wraps(fn)
    def grad_fn(*args, **kwargs):
        with tf.GradientTape(persistent=False, watch_accessed_variables=False) as tape:
            for argnum in argnums:
                tape.watch(args[argnum])

            if has_aux:
                y, aux = fn(*args, **kwargs)
            else:
                y = fn(*args, **kwargs)

        dydx = tape.gradient(y, args)

        if has_aux:
            return (y, aux), dydx

        return y, dydx

    return grad_fn
