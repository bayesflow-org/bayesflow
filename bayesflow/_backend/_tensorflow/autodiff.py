from functools import wraps

import tensorflow as tf


def grad(fn, argnums=0, has_aux=False):
    grad_fn = value_and_grad(fn, argnums=argnums, has_aux=has_aux)

    @wraps(fn)
    def wrapper(*args, **kwargs):
        val, dy = grad_fn(*args, **kwargs)

        if has_aux:
            _, aux = val
            return dy, aux
        return dy

    return wrapper


def value_and_grad(fn, argnums=0, has_aux=False):
    single_argnum = isinstance(argnums, int)
    argnums = (argnums,) if single_argnum else tuple(argnums)

    @wraps(fn)
    def grad_fn(*args, **kwargs):
        primals = tuple(args[i] for i in argnums)

        with tf.GradientTape(watch_accessed_variables=False) as tape:
            # Handles nested tensor arguments too.
            for primal in tf.nest.flatten(primals):
                tape.watch(primal)

            if has_aux:
                y, aux = fn(*args, **kwargs)
            else:
                y = fn(*args, **kwargs)

            # JAX/Torch grad require a scalar output.
            if y.shape.rank is not None and y.shape.rank != 0:
                raise ValueError(f"grad requires fn to return a scalar tensor, but got shape {y.shape}")

        dydx = tape.gradient(
            y,
            primals,
            unconnected_gradients=tf.UnconnectedGradients.ZERO,
        )

        # argnums=0 returns a tensor.
        # argnums=(0,) returns a one-element tuple.
        if single_argnum:
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

    tangents_out = acc.jvp(
        primals_out,
        unconnected_gradients=tf.UnconnectedGradients.ZERO,
    )

    if has_aux:
        return primals_out, tangents_out, aux

    return primals_out, tangents_out


def vjp(fn, *primals, has_aux=False):
    primals = tuple(primals)

    with tf.GradientTape(
        persistent=True,
        watch_accessed_variables=False,
    ) as tape:
        for primal in tf.nest.flatten(primals):
            tape.watch(primal)

        result = fn(*primals)

        if has_aux:
            y, aux = result
        else:
            y = result

    def vjp_fn(cotangents):
        return tape.gradient(
            y,
            primals,
            output_gradients=cotangents,
            unconnected_gradients=tf.UnconnectedGradients.ZERO,
        )

    if has_aux:
        return y, vjp_fn, aux

    return y, vjp_fn


def jacfwd(fn, argnums=0, has_aux=False):
    single_argnum = isinstance(argnums, int)
    argnums = (argnums,) if single_argnum else tuple(argnums)

    if not argnums:
        raise ValueError("argnums must not be empty")

    @wraps(fn)
    def jacobian_fn(*args, **kwargs):
        # Resolve negative indices and validate them.
        resolved_argnums = tuple(index if index >= 0 else len(args) + index for index in argnums)

        if any(index < 0 or index >= len(args) for index in resolved_argnums):
            raise IndexError(f"argnums={argnums} is invalid for {len(args)} arguments")

        if len(set(resolved_argnums)) != len(resolved_argnums):
            raise ValueError("argnums must not contain duplicate arguments")

        primals = tuple(args[index] for index in resolved_argnums)

        for _, primal in zip(resolved_argnums, primals):
            if tf.nest.is_nested(primal):
                raise NotImplementedError("Each differentiated argument must currently be a single tensor.")

        # Evaluate once for the primal output structure and aux value.
        result = fn(*args, **kwargs)

        if has_aux:
            primals_out, aux = result
        else:
            primals_out = result

        input_sizes = tf.stack([tf.size(primal) for primal in primals])
        input_offsets = tf.cumsum(input_sizes, exclusive=True)
        total_input_size = tf.reduce_sum(input_sizes)

        def directional_jvp(global_index):
            tangents = []

            for primal, size, offset in zip(
                primals,
                tf.unstack(input_sizes),
                tf.unstack(input_offsets),
            ):
                local_index = global_index - offset

                # Out-of-range one_hot indices produce zero vectors,
                # so only the active primal receives a basis tangent.
                tangent = tf.one_hot(
                    local_index,
                    depth=size,
                    dtype=primal.dtype,
                )
                tangent = tf.reshape(tangent, tf.shape(primal))
                tangents.append(tangent)

            with tf.autodiff.ForwardAccumulator(primals, tuple(tangents)) as accumulator:
                directional_result = fn(*args, **kwargs)

                if has_aux:
                    directional_out, _ = directional_result
                else:
                    directional_out = directional_result

            return accumulator.jvp(
                directional_out,
                unconnected_gradients=tf.UnconnectedGradients.ZERO,
            )

        # Every output leaf initially has shape:
        # (total_input_size, *output_shape)
        batched_jvps = tf.vectorized_map(
            directional_jvp,
            tf.range(total_input_size),
        )

        output_leaves = tf.nest.flatten(primals_out)
        jvp_leaves = tf.nest.flatten(batched_jvps)

        jacobian_leaves = []

        for output_leaf, leaf_jvps in zip(output_leaves, jvp_leaves):
            # Split the combined input basis back into one block per argnum.
            blocks = tf.split(leaf_jvps, input_sizes, axis=0)

            jacobians_for_output = []

            for block, primal in zip(blocks, primals):
                # (input_size, *output_shape)
                # -> (*output_shape, input_size)
                const = tf.constant([0], dtype=tf.int32)
                permutation = tf.concat([tf.range(1, tf.rank(block)), const], axis=0)
                block = tf.transpose(block, permutation)

                # (*output_shape, input_size)
                # -> (*output_shape, *input_shape)
                jacobian = tf.reshape(block, tf.concat([tf.shape(output_leaf), tf.shape(primal)], axis=0))

                jacobians_for_output.append(jacobian)

            if single_argnum:
                jacobian_leaves.append(jacobians_for_output[0])
            else:
                # For argnums=(0,), preserve the one-element tuple.
                jacobian_leaves.append(tuple(jacobians_for_output))

        # Output structure is outermost, matching JAX and Torch.
        jacobians = tf.nest.pack_sequence_as(primals_out, jacobian_leaves)

        if has_aux:
            return jacobians, aux

        return jacobians

    return jacobian_fn


def jacrev(fn, argnums=0, has_aux=False):
    single_argnum = isinstance(argnums, int)
    argnums = (argnums,) if single_argnum else tuple(argnums)

    @wraps(fn)
    def jacobian_fn(*args, **kwargs):
        primals = tuple(args[i] for i in argnums)

        with tf.GradientTape(persistent=True, watch_accessed_variables=False) as tape:
            tape.watch(tf.nest.flatten(primals))

            result = fn(*args, **kwargs)

            if has_aux:
                primals_out, aux = result
            else:
                primals_out = result

        def compute_leaf_jacobian(output_leaf):
            jacobians = tape.jacobian(  # noqa: F821
                output_leaf, primals, unconnected_gradients=tf.UnconnectedGradients.ZERO
            )

            # argnums=0 returns a Jacobian directly.
            # argnums=(0,) preserves the one-element tuple.
            if single_argnum:
                return jacobians[0]

            return tuple(jacobians)

        # Preserves the output tree as the outer structure.
        jacobians = tf.nest.map_structure(compute_leaf_jacobian, primals_out)

        if has_aux:
            return jacobians, aux

        return jacobians

    return jacobian_fn
