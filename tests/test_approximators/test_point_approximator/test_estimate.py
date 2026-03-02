import keras


def test_estimate(point_approximator, simulator, batch_size, adapter):
    num_batches = 4
    data = simulator.sample((num_batches * batch_size,))

    batch = adapter(data)
    batch = keras.tree.map_structure(keras.ops.convert_to_tensor, batch)
    batch_shapes = keras.tree.map_structure(keras.ops.shape, batch)
    point_approximator.build(batch_shapes)

    estimates = point_approximator.estimate(data)

    assert isinstance(estimates, dict)
    print(keras.tree.map_structure(keras.ops.shape, estimates))
