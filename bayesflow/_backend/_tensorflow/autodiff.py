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
    with tf.autodiff.ForwardAccumulator(primals, tangents) as acc:
        if has_aux:
            primals_out, aux = fn(*primals)
            out = primals_out, aux
        else:
            primals_out = fn(*primals)
            out = primals_out

    tangents_out = acc.jvp(primals_out)
    return out, tangents_out


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


def _get_arg_indices(argnums):
    if isinstance(argnums, int):
        return [argnums]
    return list(argnums)


def _pack_args(args, argnums_indices):
    return [args[i] for i in argnums_indices]


def jacfwd(fn, argnums=0, has_aux=False):
    def jacobian_fn(*args):
        argnums_indices = _get_arg_indices(argnums)

        # 1. Run the primal to get output shapes and aux data
        # We wrap in a function to handle the has_aux logic consistently
        out_all = fn(*args)
        if has_aux:
            primals_out, aux = out_all
        else:
            primals_out = out_all
            aux = None

        # Check if function returns multiple outputs (tuple of tensors)
        multiple_outputs = isinstance(primals_out, tuple)

        # 2. Define the vectorized JVP logic
        def get_vjp_for_arg(arg_idx):
            target_arg = args[arg_idx]
            target_shape = tf.shape(target_arg)
            n_elements = tf.reduce_prod(target_shape)

            # Create a standard basis for the input
            # (n_elements, *target_shape)
            basis = tf.eye(n_elements, dtype=target_arg.dtype)
            basis = tf.reshape(basis, tf.concat([[n_elements], target_shape], axis=0))

            def scan_jvp(tangent):
                # Replace the specific arg with its perturbed version
                new_args = list(args)

                with tf.autodiff.ForwardAccumulator(target_arg, tangent) as acc:
                    out = fn(*new_args)
                    if has_aux:
                        out = out[0]

                # jvp is the tangent of the output
                return acc.jvp(out)

            # Vectorize over the basis
            # Result shape: (n_elements, *output_shape) for single output
            # or tuple of such arrays for multiple outputs
            jaco = tf.vectorized_map(scan_jvp, basis)

            # Handle multiple outputs case
            if multiple_outputs:
                # jaco is now a tuple of arrays, each with shape (n_elements, *out_shape_i)
                result_jacobians = []
                for jaco_single in jaco:
                    output_shape = tf.shape(primals_out[len(result_jacobians)])
                    output_rank = len(primals_out[len(result_jacobians)].shape)
                    perm = list(range(1, output_rank + 1)) + [0]
                    jaco_single = tf.transpose(jaco_single, perm=perm)
                    reshaped = tf.reshape(jaco_single, tf.concat([output_shape, target_shape], axis=0))
                    result_jacobians.append(reshaped)
                return tuple(result_jacobians)
            else:
                # Move n_elements to the end and reshape to (output_shape, input_shape)
                # JAX convention: jacobian[i, j] = d(out_i) / d(in_j)
                # Result should be (out_dims..., in_dims...)
                output_rank = len(primals_out.shape)
                perm = list(range(1, output_rank + 1)) + [0]
                jaco = tf.transpose(jaco, perm=perm)
                return tf.reshape(jaco, tf.concat([tf.shape(primals_out), target_shape], axis=0))

        results = tuple(get_vjp_for_arg(i) for i in argnums_indices)

        # For multiple outputs with single argnums, we need to return a tuple of jacobians
        if multiple_outputs and isinstance(argnums, int):
            # results is a tuple with one element (the result for the single arg)
            # that element is itself a tuple of jacobians (one per output)
            out_jac = results[0]
        elif isinstance(argnums, int):
            out_jac = results[0]
        else:
            out_jac = results

        return (out_jac, aux) if has_aux else out_jac

    return jacobian_fn


def jacrev(fn, argnums=0, has_aux=False):
    def jacobian_fn(*args):
        argnums_indices = _get_arg_indices(argnums)

        # 1. Primal pass
        with tf.GradientTape(persistent=True) as tape:
            # Watch relevant arguments
            diff_args = []
            for i in argnums_indices:
                tape.watch(args[i])
                diff_args.append(args[i])

            out_all = fn(*args)
            primals_out, aux = out_all if has_aux else (out_all, None)

        # Check if function returns multiple outputs (tuple of tensors)
        multiple_outputs = isinstance(primals_out, tuple)

        # 2. Define vectorized VJP logic
        if multiple_outputs:
            # For each output, compute jacobian separately
            all_jacobians_per_arg = [[] for _ in range(len(argnums_indices))]

            for output_idx, single_output in enumerate(primals_out):
                output_shape = tf.shape(single_output)
                n_out = tf.reduce_prod(output_shape)

                # Basis for the output (n_out, *output_shape)
                output_basis = tf.eye(n_out, dtype=single_output.dtype)
                output_basis = tf.reshape(output_basis, tf.concat([[n_out], output_shape], axis=0))

                def scan_vjp(v):
                    nonlocal tape
                    return tape.gradient(single_output, diff_args, output_gradients=v)

                # Result is a list (per arg) of tensors (n_out, *arg_shape)
                jacos = tf.vectorized_map(scan_vjp, output_basis)

                # JAX contract: (out_dims..., in_dims...)
                for arg_idx, jaco_arg in enumerate(jacos):
                    arg_shape = tf.shape(args[argnums_indices[arg_idx]])
                    # Reshape from (n_out, *arg_shape) to (*out_shape, *arg_shape)
                    reshaped_jaco = tf.reshape(jaco_arg, tf.concat([output_shape, arg_shape], axis=0))
                    all_jacobians_per_arg[arg_idx].append(reshaped_jaco)

            # Convert list of jacobians to tuple of jacobians (one per output)
            final_results = []
            for arg_idx in range(len(argnums_indices)):
                # all_jacobians_per_arg[arg_idx] is a list of jacobians (one per output)
                final_results.append(tuple(all_jacobians_per_arg[arg_idx]))

            out_jac = final_results[0] if isinstance(argnums, int) else tuple(final_results)
        else:
            # Original single output logic
            output_shape = tf.shape(primals_out)
            n_out = tf.reduce_prod(output_shape)

            # Basis for the output (n_out, *output_shape)
            output_basis = tf.eye(n_out, dtype=primals_out.dtype)
            output_basis = tf.reshape(output_basis, tf.concat([[n_out], output_shape], axis=0))

            def scan_vjp(v):
                nonlocal tape
                return tape.gradient(primals_out, diff_args, output_gradients=v)

            # Result is a list (per arg) of tensors (n_out, *arg_shape)
            # vectorized_map runs the backward pass efficiently
            jacos = tf.vectorized_map(scan_vjp, output_basis)

            # JAX contract: (out_dims..., in_dims...)
            # tape.gradient returns a list if diff_args is a list
            final_results = []
            for i, jaco_arg in enumerate(jacos):
                arg_shape = tf.shape(args[argnums_indices[i]])
                # Reshape from (n_out, *arg_shape) to (*out_shape, *arg_shape)
                reshaped_jaco = tf.reshape(jaco_arg, tf.concat([output_shape, arg_shape], axis=0))
                final_results.append(reshaped_jaco)

            out_jac = tuple(final_results) if not isinstance(argnums, int) else final_results[0]

        del tape  # Explicit cleanup for persistent tape

        return (out_jac, aux) if has_aux else out_jac

    return jacobian_fn
