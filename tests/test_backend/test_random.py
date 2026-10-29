import keras
import numpy as np
import pytest


@pytest.mark.skipif(keras.backend.backend() != "jax", reason="JAX-only helper.")
def test_time_folded_uniform_varies_with_time_and_traces():
    import jax

    from bayesflow._backend._jax.random import time_folded_uniform

    key = keras.random.SeedGenerator(0).next()
    draw = jax.jit(lambda t: time_folded_uniform((3, 5), key, t))

    np.testing.assert_array_equal(draw(0.5), draw(0.5))
    assert not np.allclose(draw(0.5), draw(0.25))
