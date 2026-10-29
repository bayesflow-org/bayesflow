import jax
import keras


def time_folded_uniform(shape, seed, time):
    """Uniform samples whose key is ``seed`` folded with ``time``.

    A ``SeedGenerator`` cannot advance inside traced solver loops, so a key drawn once up front
    is combined with the solver time to get fresh randomness at every step.
    """
    key = seed if keras.ops.is_tensor(seed) else seed.next()
    time_bits = jax.lax.bitcast_convert_type(jax.numpy.ravel(time)[0].astype("float32"), "uint32")
    return jax.random.uniform(jax.random.fold_in(key, time_bits), shape)
