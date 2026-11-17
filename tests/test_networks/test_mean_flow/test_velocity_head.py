import keras
import numpy as np
import pytest

from bayesflow.experimental import MeanFlow
from bayesflow.masking import InferenceMasks

from .conftest import make_mean_flow, max_abs_diff


def test_velocity_without_head_is_boundary_average_velocity(random_samples, random_conditions):
    network = make_mean_flow(random_samples, random_conditions, velocity_head_layers=0)
    v = network.velocity(random_samples, 0.4, conditions=random_conditions)
    u = network._average_velocity(random_samples, 0.4, 0.4, random_conditions, False, InferenceMasks())
    np.testing.assert_array_equal(keras.ops.convert_to_numpy(v), keras.ops.convert_to_numpy(u))


def test_velocity_head(random_samples, random_conditions):
    network = make_mean_flow(random_samples, random_conditions, velocity_head_layers=1)
    leading_block, trailing_block = network.subnet.blocks
    (head_block,) = network.velocity_blocks

    # the head owns a clone of the trailing block
    assert type(head_block) is type(trailing_block)
    assert head_block.get_config() | {"name": None} == trailing_block.get_config() | {"name": None}
    head_weights = {id(w) for w in head_block.weights}
    assert head_weights and head_weights.isdisjoint(id(w) for w in network.subnet.weights)

    v = network.velocity(random_samples, 0.4, conditions=random_conditions)
    u = network._average_velocity(random_samples, 0.4, 0.4, random_conditions, False, InferenceMasks())
    assert max_abs_diff(v, u) > 1e-3

    # and shares the leading block with the subnet
    for weight in leading_block.weights:
        weight.assign(keras.random.normal(weight.shape))
    assert max_abs_diff(network.velocity(random_samples, 0.4, conditions=random_conditions), v) > 1e-3


def test_velocity_head_after_shared_blocks_of_other_width(random_samples, random_conditions):
    network = make_mean_flow(
        random_samples, random_conditions, subnet_kwargs=dict(widths=(8, 16)), velocity_head_layers=1
    )
    v = network.velocity(random_samples, 0.4, conditions=random_conditions)
    assert keras.ops.shape(v) == keras.ops.shape(random_samples)


def test_velocity_head_raises():
    with pytest.raises(ValueError, match="embed"):
        MeanFlow(subnet="mlp")
    with pytest.raises(ValueError, match="velocity_head_layers"):
        MeanFlow(subnet_kwargs=dict(widths=(8, 8)), velocity_head_layers=3)
