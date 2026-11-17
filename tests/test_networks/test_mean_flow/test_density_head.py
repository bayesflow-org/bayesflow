import keras
import pytest

from bayesflow.experimental import MeanFlow
from bayesflow.masking import InferenceMasks
from tests.utils import assert_allclose

from .conftest import make_mean_flow, make_transformer_mean_flow, max_abs_diff


def test_density_head_log_prob(random_samples, random_conditions):
    network = make_mean_flow(random_samples, random_conditions, density_head=True)

    # the forward pass runs from data at t = 0 to noise at t = 1
    times = keras.ops.convert_to_numpy(MeanFlow.make_time_schedule(2))[::-1]
    x, log_det = random_samples, 0.0
    for source, target in zip(times[:-1], times[1:]):
        u, divergence = network._average_velocity_and_divergence(
            x, source, target, random_conditions, False, InferenceMasks()
        )
        x = x + (target - source) * u
        log_det += (target - source) * keras.ops.sum(divergence, axis=-1)

    z, log_density = network(random_samples, conditions=random_conditions, density=True, steps=2)
    assert_allclose(z, x, atol=1e-5)
    assert_allclose(log_density, network.base_distribution.log_prob(x) + log_det, atol=1e-4)


@pytest.mark.parametrize("subnet", ["time_mlp", "diffusion_transformer"])
def test_density_head_shares_subnet_pass(subnet, random_samples, random_conditions):
    if subnet == "time_mlp":
        # a final width equal to the target dimension, where the average velocity has no projector
        network = make_mean_flow(
            random_samples, random_conditions, subnet_kwargs=dict(widths=(8, 2)), density_head=True
        )
    else:
        network = make_transformer_mean_flow(random_samples, random_conditions, density_head=True)

    u, divergence = network._average_velocity_and_divergence(
        random_samples, 0.8, 0.3, random_conditions, False, InferenceMasks()
    )
    expected = network._average_velocity(random_samples, 0.8, 0.3, random_conditions, False, InferenceMasks())
    assert_allclose(u, expected, atol=1e-6)
    assert max_abs_diff(u, divergence) > 1e-3
