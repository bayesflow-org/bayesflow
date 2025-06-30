import keras
from tests.utils import check_combination_simulator_adapter, check_approximator_multivariate_normal_score


def test_approximator_sample(approximator, simulator, batch_size, adapter):
    check_combination_simulator_adapter(simulator, adapter)
    # as long as MultivariateNormalScore is unstable, skip
    check_approximator_multivariate_normal_score(approximator)

    num_batches = 4
    data = simulator.sample((num_batches * batch_size,))

    batch = adapter(data)
    batch = keras.tree.map_structure(keras.ops.convert_to_tensor, batch)
    batch_shapes = keras.tree.map_structure(keras.ops.shape, batch)
    approximator.build(batch_shapes)

    samples = approximator.sample(num_samples=2, conditions=data)

    assert isinstance(samples, dict)
