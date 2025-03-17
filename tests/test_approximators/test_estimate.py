import keras


def test_approximator_estimate(approximator, simulator, batch_size, adapter):
    approximator = approximator

    num_batches = 4
    data = simulator.sample((num_batches * batch_size,))

    batch = adapter(data)
    approximator.build_from_data(batch)

    estimates = approximator.estimate(data)

    assert isinstance(estimates, dict)
    print(keras.tree.map_structure(keras.ops.shape, estimates))
