from functools import wraps

import torch


def grad(fn, argnums=0, has_aux=False):
    return torch.func.grad(fn, argnums=argnums, has_aux=has_aux)


def value_and_grad(fn, argnums=0, has_aux=False):
    grad_fn = torch.func.grad_and_value(fn, argnums=argnums, has_aux=has_aux)

    @wraps(fn)
    def wrapper(*args, **kwargs):
        dydx, y = grad_fn(*args, **kwargs)
        return y, dydx

    return wrapper


def jvp(fn, primals, tangents, has_aux=False):
    return torch.func.jvp(fn, primals, tangents, has_aux=has_aux)


def vjp(fn, *primals, has_aux=False):
    return torch.func.vjp(fn, *primals, has_aux=has_aux)


def jacfwd(fn, argnums=0, has_aux=False):
    return torch.func.jacfwd(fn, argnums=argnums, has_aux=has_aux)


def jacrev(fn, argnums=0, has_aux=False):
    return torch.func.jacrev(fn, argnums=argnums, has_aux=has_aux)
