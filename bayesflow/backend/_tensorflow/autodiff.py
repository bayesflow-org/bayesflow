from functools import wraps

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
        with tf.GradientTape(persistent=False, watch_accessed_variables=False) as tape:
            for argnum in argnums:
                tape.watch(args[argnum])

            if has_aux:
                y, aux = fn(*args, **kwargs)
            else:
                y = fn(*args, **kwargs)

        dydx = tape.gradient(y, args)

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
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            for argnum in argnums:
                tape.watch(args[argnum])

            if has_aux:
                y, aux = fn(*args, **kwargs)
            else:
                y = fn(*args, **kwargs)

        sources = [args[i] for i in argnums]
        jacs = tape.jacobian(y, sources)

        if len(argnums) == 1:
            # follow the jax way to return the jacobian directly if only one argument is differentiated
            jacs = jacs[0]
        else:
            jacs = tuple(jacs)

        return (jacs, aux) if has_aux else jacs

    return wrapper


def jacfwd(fn, argnums=0, has_aux=False):
    if isinstance(argnums, int):
        argnums = [argnums]

    @wraps(fn)
    def wrapper(*args, **kwargs):
        jacs = []
        for argnum in argnums:
            p = args[argnum]
            p_shape = tf.shape(p)
            n = tf.reduce_prod(p_shape)

            def get_jvp(i):
                # Construct standard basis vector and reshape to primal shape
                tangent = tf.reshape(tf.one_hot(i, n, dtype=p.dtype), p_shape)
                tangents = tuple(tf.zeros_like(a) for a in args)
                tangents = list(tangents)
                tangents[argnum] = tangent

                with tf.autodiff.ForwardAccumulator(args, tuple(tangents)) as acc:
                    out = fn(*args, **kwargs)
                    y_val = out[0] if has_aux else out
                return acc.jvp(y_val)

            # Parallelize JVP computation over the input basis
            jvp_stack = tf.vectorized_map(get_jvp, tf.range(n))

            # Reshape and transpose to match (output_shape, input_shape)
            # jvp_stack is [input_dim, output_dims...] -> move input_dim to end
            perm = tf.roll(tf.range(tf.rank(jvp_stack)), shift=-1, axis=0)
            jac = tf.transpose(jvp_stack, perm)
            jacs.append(tf.reshape(jac, tf.concat([tf.shape(jac)[:-1], p_shape], axis=0)))

        if has_aux:
            _, aux = fn(*args, **kwargs)

        jacs = jacs[0] if len(argnums) == 1 else tuple(jacs)
        return (jacs, aux) if has_aux else jacs

    return wrapper
