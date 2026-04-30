import jax
import keras


if keras.config.is_nnx_enabled():
    import flax.nnx as nnx

    def jit(fn):
        return nnx.jit(fn)
else:

    def jit(fn):
        return jax.jit(fn)
