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
            dydx = dydx[0]

        if has_aux:
            return (y, aux), dydx

        return y, dydx

    return grad_fn


def jvp(fn, primals, tangents, has_aux=False):
    primals = tuple(primals)
    tangents = tuple(tangents)

    with tf.autodiff.ForwardAccumulator(primals, tangents) as acc:
        if has_aux:
            primals_out, aux = fn(*primals)
        else:
            primals_out = fn(*primals)

    tangents_out = acc.jvp(primals_out)

    if has_aux:
        return primals_out, tangents_out, aux

    return primals_out, tangents_out


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


def jacfwd(fn, argnums=0, has_aux=False):
    if isinstance(argnums, int):
        argnums_list = [argnums]
    else:
        argnums_list = list(argnums)

    def jacobian_fn(*args):
        with tf.GradientTape(persistent=True, watch_accessed_variables=False) as tape:
            for i in argnums_list:
                tape.watch(args[i])

            out_all = fn(*args)
            if has_aux:
                primals_out, aux = out_all
            else:
                primals_out = out_all
                aux = None

        jacs = []

        for arg_idx in argnums_list:
            target_arg = args[arg_idx]

            if not isinstance(primals_out, tuple):
                # single output, compute jacobian directly
                jac = tape.jacobian(primals_out, target_arg)
                jacs.append(jac)
            else:
                # multiple outputs, compute Jacobians individually
                output_jacs = []
                for output_idx, single_output in enumerate(primals_out):
                    jac = tape.jacobian(single_output, target_arg)
                    output_jacs.append(jac)
                jacs.append(tuple(output_jacs))

        if isinstance(argnums, int):
            jacs = jacs[0]
        else:
            jacs = tuple(jacs)

        return (jacs, aux) if has_aux else jacs

    return jacobian_fn


def jacrev(fn, argnums=0, has_aux=False):
    if isinstance(argnums, int):
        argnums_list = [argnums]
    else:
        argnums_list = list(argnums)

    def jacobian_fn(*args):
        with tf.GradientTape(persistent=True) as tape:
            diff_args = []
            for i in argnums_list:
                tape.watch(args[i])
                diff_args.append(args[i])

            out_all = fn(*args)
            primals_out, aux = out_all if has_aux else (out_all, None)

        if not isinstance(primals_out, tuple):
            # single output
            output_shape = tf.shape(primals_out)
            n_out = tf.reduce_prod(output_shape)

            # basis for the output (n_out, *output_shape)
            output_basis = tf.eye(n_out, dtype=primals_out.dtype)
            output_basis = tf.reshape(output_basis, tf.concat([[n_out], output_shape], axis=0))

            def scan_vjp(v):
                nonlocal tape
                return tape.gradient(primals_out, diff_args, output_gradients=v)

            # use vmap to run the backward pass efficiently
            # the result is a list (per arg) of tensors (n_out, *arg_shape)
            jacs = tf.vectorized_map(scan_vjp, output_basis)

            # jax contract: (out_dims..., in_dims...)
            # tape.gradient returns a list if diff_args is a list
            final_results = []
            for i, jaco_arg in enumerate(jacs):
                arg_shape = tf.shape(args[argnums_list[i]])
                reshaped_jac = tf.reshape(jaco_arg, tf.concat([output_shape, arg_shape], axis=0))
                final_results.append(reshaped_jac)

            jacs = tuple(final_results) if not isinstance(argnums, int) else final_results[0]
        else:
            # multiple outputs, compute Jacobians individually
            all_jacobians_per_arg = [[] for _ in range(len(argnums_list))]

            for output_idx, single_output in enumerate(primals_out):
                output_shape = tf.shape(single_output)
                n_out = tf.reduce_prod(output_shape)

                # basis vectors for the vjp (n_out, *output_shape)
                output_basis = tf.eye(n_out, dtype=single_output.dtype)
                output_basis = tf.reshape(output_basis, tf.concat([[n_out], output_shape], axis=0))

                def scan_vjp(v):
                    nonlocal tape
                    return tape.gradient(single_output, diff_args, output_gradients=v)

                # use vmap to run the backward pass efficiently
                # the result is a list (per arg) of tensors (n_out, *arg_shape)
                jacos = tf.vectorized_map(scan_vjp, output_basis)

                # jax contract: (out_dims..., in_dims...)
                for arg_idx, jaco_arg in enumerate(jacos):
                    arg_shape = tf.shape(args[argnums_list[arg_idx]])
                    # Reshape from (n_out, *arg_shape) to (*out_shape, *arg_shape)
                    reshaped_jaco = tf.reshape(jaco_arg, tf.concat([output_shape, arg_shape], axis=0))
                    all_jacobians_per_arg[arg_idx].append(reshaped_jaco)

            # convert list of jacobians to tuple of jacobians (one per output)
            final_results = []
            for arg_idx in range(len(argnums_list)):
                # all_jacobians_per_arg[arg_idx] is a list of jacobians (one per output)
                final_results.append(tuple(all_jacobians_per_arg[arg_idx]))

            if isinstance(argnums, int):
                jacs = final_results[0]
            else:
                jacs = tuple(final_results)

        return (jacs, aux) if has_aux else jacs

    return jacobian_fn
