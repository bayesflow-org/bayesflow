import keras


def test_expand_to_target_rank():
    from bayesflow.experimental.graphical_approximator.tensor_concatenation import expand_to_target_rank

    assert expand_to_target_rank(keras.random.normal((2, 3)), 5).shape == (2, 1, 1, 1, 3)
    assert expand_to_target_rank(keras.random.normal((2, 3, 4)), 3).shape == (2, 3, 4)


def test_concatenate():
    from bayesflow.experimental.graphical_approximator.tensor_concatenation import concatenate

    x = keras.random.normal((20, 5))
    y = keras.random.normal((20, 15, 3))
    z = concatenate([x, y])
    assert z.shape == (20, 15, 8)

    x = keras.random.normal((20, 1))
    y = keras.random.normal((20, 15, 3))
    z = concatenate([x, y])
    assert z.shape == (20, 15, 4)

    x = keras.Input(shape=(5,))
    y = keras.Input(shape=(15, 3))
    z = concatenate([x, y])
    model = keras.Model([x, y], z)
    out = model([keras.ops.ones((20, 5)), keras.ops.ones((20, 15, 3))])
    assert out.shape == (20, 15, 8)
