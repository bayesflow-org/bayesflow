import keras
import numpy as np
import pytest

from bayesflow.utils.serialization import serialize, deserialize

from tests.utils import assert_allclose, assert_layers_equal


def test_build(inference_network, random_samples, random_conditions):
    assert inference_network.built is False

    samples_shape = keras.ops.shape(random_samples)
    conditions_shape = keras.ops.shape(random_conditions) if random_conditions is not None else None

    inference_network.build(samples_shape, conditions_shape=conditions_shape)

    assert inference_network.built is True

    # check the model has variables
    assert inference_network.variables, "Model has no variables."


def test_variable_batch_size(inference_network, random_samples, random_conditions):
    # build with one batch size
    samples_shape = keras.ops.shape(random_samples)
    conditions_shape = keras.ops.shape(random_conditions) if random_conditions is not None else None
    inference_network.build(samples_shape, conditions_shape=conditions_shape)

    # run with another batch size
    batch_sizes = np.random.choice(10, replace=False, size=3)
    for bs in batch_sizes:
        new_input = keras.ops.zeros((bs,) + keras.ops.shape(random_samples)[1:])
        if random_conditions is None:
            new_conditions = None
        else:
            new_conditions = keras.ops.zeros((bs,) + keras.ops.shape(random_conditions)[1:])

        try:
            inference_network(new_input, conditions=new_conditions)
        except NotImplementedError:
            # network is not invertible
            pass
        inference_network(new_input, conditions=new_conditions, inverse=True)


@pytest.mark.parametrize("density", [True, False])
def test_output_structure(density, generative_inference_network, random_samples, random_conditions):
    try:
        output = generative_inference_network(random_samples, conditions=random_conditions, density=density)
    except NotImplementedError:
        # network not invertible
        return

    if density:
        assert isinstance(output, tuple)
        assert len(output) == 2

        forward_output, forward_log_density = output

        assert keras.ops.is_tensor(forward_output)
        assert keras.ops.is_tensor(forward_log_density)
    else:
        assert keras.ops.is_tensor(output)


def test_output_shape(generative_inference_network, random_samples, random_conditions):
    try:
        forward_output, forward_log_density = generative_inference_network(
            random_samples, conditions=random_conditions, density=True
        )
    except NotImplementedError:
        # network is not invertible, not forward function available
        return

    assert keras.ops.shape(forward_output) == keras.ops.shape(random_samples)
    assert keras.ops.shape(forward_log_density) == (keras.ops.shape(random_samples)[0],)

    inverse_output, inverse_log_density = generative_inference_network(
        random_samples, conditions=random_conditions, density=True, inverse=True
    )

    assert keras.ops.shape(inverse_output) == keras.ops.shape(random_samples)
    assert keras.ops.shape(inverse_log_density) == (keras.ops.shape(random_samples)[0],)


def test_cycle_consistency(generative_inference_network, random_samples, random_conditions):
    # cycle-consistency means the forward and inverse methods are inverses of each other
    import bayesflow as bf

    if isinstance(generative_inference_network, bf.experimental.DiffusionModel):
        pytest.skip(reason="test unstable for untrained diffusion models")
    try:
        forward_output, forward_log_density = generative_inference_network(
            random_samples, conditions=random_conditions, density=True
        )
    except NotImplementedError:
        pytest.skip(reason="network is not invertible")
    inverse_output, inverse_log_density = generative_inference_network(
        forward_output, conditions=random_conditions, density=True, inverse=True
    )

    assert_allclose(random_samples, inverse_output, atol=1e-3, rtol=1e-3)
    assert_allclose(forward_log_density, inverse_log_density, atol=1e-3, rtol=1e-3)


def test_density_numerically(generative_inference_network, random_samples, random_conditions):
    from bayesflow.utils import jacobian

    try:
        output, log_density = generative_inference_network(random_samples, conditions=random_conditions, density=True)
    except NotImplementedError:
        # network does not support density estimation
        return

    def f(x):
        return generative_inference_network(x, conditions=random_conditions)

    numerical_output, numerical_jacobian = jacobian(f, random_samples, return_output=True)

    # output should be identical, otherwise this test does not work (e.g. for stochastic networks)
    assert_allclose(output, numerical_output)

    log_prob = generative_inference_network.base_distribution.log_prob(output)

    # use change of variables to compute the numerical log density
    numerical_log_density = log_prob + keras.ops.log(keras.ops.abs(keras.ops.det(numerical_jacobian)))

    # use a high tolerance because the numerical jacobian is not very accurate
    assert_allclose(log_density, numerical_log_density, rtol=1e-3, atol=1e-3)


def test_serialize_deserialize(inference_network, random_samples, random_conditions):
    # to save, the model must be built
    xz_shape = keras.ops.shape(random_samples)
    conditions_shape = keras.ops.shape(random_conditions) if random_conditions is not None else None
    inference_network.build(xz_shape, conditions_shape)

    serialized = serialize(inference_network)
    deserialized = deserialize(serialized)
    reserialized = serialize(deserialized)

    assert keras.tree.lists_to_tuples(serialized) == keras.tree.lists_to_tuples(reserialized)


def test_save_and_load(tmp_path, inference_network, random_samples, random_conditions):
    # to save, the model must be built
    xz_shape = keras.ops.shape(random_samples)
    conditions_shape = keras.ops.shape(random_conditions) if random_conditions is not None else None
    inference_network.build(xz_shape, conditions_shape)

    keras.saving.save_model(inference_network, tmp_path / "model.keras")
    loaded = keras.saving.load_model(tmp_path / "model.keras")

    assert_layers_equal(inference_network, loaded)


def test_compute_metrics(inference_network, random_samples, random_conditions):
    xz_shape = keras.ops.shape(random_samples)
    conditions_shape = keras.ops.shape(random_conditions) if random_conditions is not None else None
    inference_network.build(xz_shape, conditions_shape)

    metrics = inference_network.compute_metrics(random_samples, conditions=random_conditions)
    assert "loss" in metrics
