import jax


def grad(fn, argnums=0, has_aux=False):
    return jax.grad(fn, argnums=argnums, has_aux=has_aux)


def value_and_grad(fn, argnums=0, has_aux=False):
    return jax.value_and_grad(fn, argnums=argnums, has_aux=has_aux)


def jvp(fn, primals, tangents, has_aux=False):
    return jax.jvp(fn, primals, tangents, has_aux=has_aux)


def vjp(fn, *primals, has_aux=False):
    return jax.vjp(fn, *primals, has_aux=has_aux)


def jacfwd(fn, argnums=0, has_aux=False):
    return jax.jacfwd(fn, argnums=argnums, has_aux=has_aux)


def jacrev(fn, argnums=0, has_aux=False):
    return jax.jacrev(fn, argnums=argnums, has_aux=has_aux)
