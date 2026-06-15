import keras
import pytest

from bayesflow.experimental.graphical_approximator.non_exchangeable_wrapper import NonExchangeableWrapper
from bayesflow.networks import CouplingFlow, DeepSet


@pytest.fixture
def built_approximator(crossed_design_irt_simulator, crossed_design_irt_approximator):
    data = crossed_design_irt_simulator.sample(2)
    adapted = crossed_design_irt_approximator.adapter(data)
    data_shapes = crossed_design_irt_approximator._data_shapes(adapted)
    crossed_design_irt_approximator.build(data_shapes)
    crossed_design_irt_approximator.compile()
    crossed_design_irt_approximator.compute_metrics(**adapted)

    return crossed_design_irt_approximator, adapted


def test_serialization(crossed_design_irt_approximator):
    wrapper = crossed_design_irt_approximator.inference_networks[1]
    config = wrapper.get_config()
    restored = NonExchangeableWrapper.from_config(config)
    assert isinstance(restored, NonExchangeableWrapper)


def test_build():
    wrapper = NonExchangeableWrapper(CouplingFlow(), DeepSet(summary_dim=8))
    wrapper.build((2, 4, 2))
    assert wrapper.built

    wrapper.build((2, 4, 2))  # early return for already-built wrapper
    wrapper2 = NonExchangeableWrapper(CouplingFlow(), DeepSet(summary_dim=8))
    wrapper2.build((2, 4, 2), conditions_shape=(2, 4, 12))
    assert wrapper2.built


def test_call(built_approximator):
    from bayesflow.experimental.graphical_approximator.inference_conditions import inference_conditions_by_network
    from bayesflow.experimental.graphical_approximator.inference_variables import inference_variables_by_network

    approximator, adapted = built_approximator
    wrapper = approximator.inference_networks[1]
    variables = inference_variables_by_network(approximator, adapted)
    conditions = inference_conditions_by_network(approximator, adapted, approximator.summary_registry)
    xz, cond = variables[1], conditions[1]
    noise = keras.ops.zeros_like(xz)

    result = wrapper(xz, conditions=cond, inverse=False, density=False)
    assert result.shape == xz.shape

    z, _ = wrapper(xz, conditions=cond, inverse=False, density=True)
    assert z.shape == xz.shape

    result = wrapper(noise, conditions=cond, inverse=True, density=False)
    assert result.shape == xz.shape

    values, _ = wrapper(noise, conditions=cond, inverse=True, density=True)
    assert values.shape == xz.shape

    samples = wrapper.sample(tuple(xz.shape)[:-1], conditions=cond)
    assert samples.shape == xz.shape

    log_prob = wrapper.log_prob(xz, conditions=cond)
    assert log_prob is not None


def test_prepare_inference_conditions(built_approximator):
    from bayesflow.experimental.graphical_approximator.inference_conditions import inference_conditions_by_network
    from bayesflow.experimental.graphical_approximator.inference_variables import inference_variables_by_network

    approximator, adapted = built_approximator
    wrapper = approximator.inference_networks[1]
    variables = inference_variables_by_network(approximator, adapted)
    conditions = inference_conditions_by_network(approximator, adapted, approximator.summary_registry)
    xz = variables[1]

    result = wrapper._prepare_inference_conditions(xz, i=0, conditions=None)
    assert result is not None

    single_step_cond = conditions[1][..., :1, :]
    result = wrapper._prepare_inference_conditions(xz, i=0, conditions=single_step_cond)
    assert result is not None
