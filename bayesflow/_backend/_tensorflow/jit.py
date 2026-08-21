import tensorflow as tf


def jit(fn):
    return tf.function(fn, jit_compile=True)
