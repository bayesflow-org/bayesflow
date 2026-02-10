import keras

from functools import singledispatch


@singledispatch
def find_pooling(arg, *args, **kwargs):
    raise TypeError(f"Cannot infer pooling from {arg!r}.")


@find_pooling.register
def _(name: str, *args, **kwargs):
    match name.lower():
        case "mean" | "avg" | "average":
            pooling = keras.layers.Lambda(lambda inp: keras.ops.mean(inp, axis=-2))
        case "max":
            pooling = keras.layers.Lambda(lambda inp: keras.ops.max(inp, axis=-2))
        case "min":
            pooling = keras.layers.Lambda(lambda inp: keras.ops.min(inp, axis=-2))
        case "learnable" | "pma" | "attention":
            from bayesflow.networks.transformers.pma import PoolingByMultiHeadAttention

            pooling = PoolingByMultiHeadAttention(*args, **kwargs)
        case other:
            raise ValueError(f"Unsupported pooling name: '{other}'.")

    return pooling


@find_pooling.register
def _(constructor: type, *args, **kwargs):
    return constructor(*args, **kwargs)


@find_pooling.register
def _(pooling: keras.Layer, *args, **kwargs):
    return pooling
