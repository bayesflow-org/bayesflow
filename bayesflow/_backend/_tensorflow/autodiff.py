from functools import partial, wraps

import tensorflow as tf


def grad(fn, argnums=0, has_aux=False):
    grad_fn = value_and_grad(fn, argnums=argnums, has_aux=has_aux)

    @wraps(fn)
    def wrapper(*args, **kwargs):
        val, dy = grad_fn(*args, **kwargs)
        if has_aux:
            y, aux = val
            return dy, aux
        return dy

    return wrapper


def value_and_grad(fn, argnums=0, has_aux=False):
    if isinstance(argnums, int):
        argnums = [argnums]

    @wraps(fn)
    def grad_fn(*args, **kwargs):
        nonlocal fn

        fn = partial(fn, **kwargs)
        primals = [args[i] for i in argnums]

        with tf.GradientTape(persistent=False, watch_accessed_variables=False) as tape:
            for p in primals:
                tape.watch(p)

            if has_aux:
                y, aux = fn(*args, **kwargs)
            else:
                y = fn(*args, **kwargs)

        if tf.executing_eagerly():
            dydx = tape.gradient(y, primals)
        else:
            dydx = tf.gradients(y, primals)

        dydx = tuple(dydx)

        if len(argnums) == 1:
            # follow the jax way to return the gradient directly if only one argument is differentiated
            dydx = dydx[0]

        if has_aux:
            return (y, aux), dydx

        return y, dydx

    return grad_fn


def jvp(fn, primals, tangents, has_aux=False):
    with tf.autodiff.ForwardAccumulator(primals, tangents) as acc:
        if has_aux:
            y, aux = fn(*primals)
            out = y, aux
        else:
            y = fn(*primals)
            out = y
    _jvp = acc.jvp(y)
    return out, _jvp


def vjp(fn, *primals, has_aux=False) -> tuple:
    # use a persistent tape so the vjp_fn can be called multiple times
    with tf.GradientTape(persistent=True, watch_accessed_variables=False) as tape:
        for p in primals:
            tape.watch(p)

        if has_aux:
            y, aux = fn(*primals)
            out = y, aux
        else:
            y = fn(*primals)
            out = y

    def vjp_fn(cotangent):
        return tape.gradient(y, primals, output_gradients=cotangent)

    return out, vjp_fn


def jacrev(fn, argnums=0, has_aux=False):
    if isinstance(argnums, int):
        argnums = [argnums]

    @wraps(fn)
    def wrapper(*args, **kwargs):
        with tf.GradientTape(persistent=False, watch_accessed_variables=False) as tape:
            for argnum in argnums:
                tape.watch(args[argnum])

            if has_aux:
                out, aux = fn(*args, **kwargs)
            else:
                out = fn(*args, **kwargs)

        jacs = tape.jacobian(target=out, sources=args)

        if len(argnums) == 1:
            return jacs[0]

        return jacs

    return wrapper


def jacfwd(fn, argnums=0, has_aux=False):
    if isinstance(argnums, int):
        argnums = [argnums]

    @wraps(fn)
    def jac_fn(*args, **kwargs):
        bound_fn = partial(fn, **kwargs)

        primals = [args[i] for i in argnums]
        shapes = [tf.shape(p) for p in primals]
        sizes = [tf.reduce_prod(s) for s in shapes]

        def jvp_fn(tangent):
            tangents = [tf.reshape(t, s) for t, s in zip(tf.split(tangent, sizes), shapes)]
            out, _jvp = jvp(bound_fn, primals, tangents, has_aux=has_aux)
            if has_aux:
                _, aux = out
                return _jvp, aux
            return _jvp, None

        eye = tf.eye(sum(sizes))
        jacobian, aux = tf.vectorized_map(jvp_fn, eye)
        jacobian = tf.squeeze(jacobian)

        if has_aux:
            return tf.transpose(jacobian), tf.nest.map_structure(lambda x: x[0], aux)
        return tf.transpose(jacobian)

    return jac_fn
