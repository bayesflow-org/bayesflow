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
        resolved_argnums = tuple(index if index >= 0 else len(args) + index for index in argnums)

        if any(index < 0 or index >= len(args) for index in resolved_argnums):
            raise IndexError(f"argnums={argnums} is invalid for {len(args)} arguments")

        if len(set(resolved_argnums)) != len(resolved_argnums):
            raise ValueError("argnums must not contain duplicates")

        primals = tuple(args[index] for index in resolved_argnums)

        for primal in primals:
            if tf.nest.is_nested(primal):
                raise NotImplementedError("Nested differentiated arguments are not supported.")

        input_sizes = tf.stack([tf.size(primal) for primal in primals])
        input_offsets = tf.cumsum(
            input_sizes,
            exclusive=True,
        )
        total_input_size = tf.reduce_sum(input_sizes)

        sizes = tf.unstack(input_sizes)
        offsets = tf.unstack(input_offsets)

        def directional_jvp(global_index):
            tangents = []

            for primal, size, offset in zip(
                primals,
                sizes,
                offsets,
            ):
                local_index = global_index - offset

                # An out-of-range one-hot index produces all zeros,
                # so exactly one selected argument receives a basis
                # tangent for each global index.
                tangent = tf.one_hot(
                    local_index,
                    depth=size,
                    dtype=primal.dtype,
                )
                tangent = tf.reshape(
                    tangent,
                    tf.shape(primal),
                )
                tangents.append(tangent)

            with tf.autodiff.ForwardAccumulator(
                primals,
                tuple(tangents),
            ) as accumulator:
                result = fn(*args, **kwargs)

                if has_aux:
                    directional_out, aux = result
                else:
                    directional_out = result

            jvp_out = accumulator.jvp(
                directional_out,
                unconnected_gradients=tf.UnconnectedGradients.ZERO,
            )

            if has_aux:
                return jvp_out, aux

            return jvp_out

        # No separate primal-only call.
        batched = tf.vectorized_map(
            directional_jvp,
            tf.range(total_input_size),
        )

        if has_aux:
            batched_jvps, batched_aux = batched

            # Aux is identical for every basis direction.
            aux = tf.nest.map_structure(
                lambda leaf: leaf[0],
                batched_aux,
            )
        else:
            batched_jvps = batched

        jacobian_leaves = []

        for leaf_jvps in tf.nest.flatten(batched_jvps):
            # leaf_jvps:
            # (total_input_size, *output_shape)
            blocks = tf.split(
                leaf_jvps,
                input_sizes,
                axis=0,
            )

            jacobians_for_output = []

            for block, primal in zip(blocks, primals):
                # (input_size, *output_shape)
                # -> (*output_shape, input_size)
                permutation = tf.concat(
                    [
                        tf.range(1, tf.rank(block)),
                        tf.constant([0], dtype=tf.int32),
                    ],
                    axis=0,
                )
                block = tf.transpose(
                    block,
                    permutation,
                )

                # (*output_shape, input_size)
                # -> (*output_shape, *input_shape)
                jacobian = tf.reshape(
                    block,
                    tf.concat(
                        [
                            tf.shape(leaf_jvps)[1:],
                            tf.shape(primal),
                        ],
                        axis=0,
                    ),
                )
                jacobians_for_output.append(jacobian)

            if single_argnum:
                jacobian_leaves.append(jacobians_for_output[0])
            else:
                jacobian_leaves.append(tuple(jacobians_for_output))

        # Preserve the output tree as the outer structure.
        jacobians = tf.nest.pack_sequence_as(
            batched_jvps,
            jacobian_leaves,
        )

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
