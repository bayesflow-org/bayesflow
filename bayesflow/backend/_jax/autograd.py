import jax


def grad(fn, argnums=0, has_aux=False):
    return jax.grad(fn, argnums=argnums, has_aux=has_aux)


def value_and_grad(fn, argnums=0, has_aux=False):
    return jax.value_and_grad(fn, argnums=argnums, has_aux=has_aux)
