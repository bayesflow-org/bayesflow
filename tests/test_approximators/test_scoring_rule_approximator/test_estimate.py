import keras
from tests.utils import check_combination_simulator_adapter


def test_estimate(scoring_rule_approximator_any, simulator, batch_size, adapter):
    check_combination_simulator_adapter(simulator, adapter)

    num_batches = 4
    data = simulator.sample((num_batches * batch_size,))

    batch = adapter(data)
    batch = keras.tree.map_structure(keras.ops.convert_to_tensor, batch)
    batch_shapes = keras.tree.map_structure(keras.ops.shape, batch)
    scoring_rule_approximator_any.build(batch_shapes)

    estimates = scoring_rule_approximator_any.estimate(data)

    assert isinstance(estimates, dict)
    print(keras.tree.map_structure(keras.ops.shape, estimates))
