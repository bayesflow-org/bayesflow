import keras
import numpy as np
import pytest
from keras.saving import (
    deserialize_keras_object as deserialize,
    serialize_keras_object as serialize,
)

from tests.utils import allclose, assert_layers_equal


def test_build(inference_network, random_samples, random_conditions):
    assert inference_network.built is False

    inference_network(random_samples, conditions=random_conditions)

    assert inference_network.built is True

    # check the model has variables
    assert inference_network.variables, "Model has no variables."


def test_variable_batch_size(inference_network, random_samples, random_conditions):
    # build with one batch size
    inference_network(random_samples, conditions=random_conditions)

    # run with another batch size
    batch_sizes = np.random.choice(10, replace=False, size=3)
    for bs in batch_sizes:
        new_input = keras.ops.zeros((bs,) + keras.ops.shape(random_samples)[1:])
        if random_conditions is None:
            new_conditions = None
        else:
            new_conditions = keras.ops.zeros((bs,) + keras.ops.shape(random_conditions)[1:])

        inference_network(new_input, conditions=new_conditions)
        inference_network(new_input, conditions=new_conditions, inverse=True)


@pytest.mark.parametrize("density", [True, False])
def test_output_structure(density, inference_network, random_samples, random_conditions):
    output = inference_network(random_samples, conditions=random_conditions, density=density)

    if density:
        assert isinstance(output, tuple)
        assert len(output) == 2

        forward_output, forward_log_density = output

        assert keras.ops.is_tensor(forward_output)
        assert keras.ops.is_tensor(forward_log_density)
    else:
        assert keras.ops.is_tensor(output)


def test_output_shape(inference_network, random_samples, random_conditions):
    forward_output, forward_log_density = inference_network(random_samples, conditions=random_conditions, density=True)

    assert keras.ops.shape(forward_output) == keras.ops.shape(random_samples)
    assert keras.ops.shape(forward_log_density) == (keras.ops.shape(random_samples)[0],)

    inverse_output, inverse_log_density = inference_network(
        random_samples, conditions=random_conditions, density=True, inverse=True
    )

    assert keras.ops.shape(inverse_output) == keras.ops.shape(random_samples)
    assert keras.ops.shape(inverse_log_density) == (keras.ops.shape(random_samples)[0],)


def test_cycle_consistency(inference_network, random_samples, random_conditions):
    # cycle-consistency means the forward and inverse methods are inverses of each other
    forward_output, forward_log_density = inference_network(random_samples, conditions=random_conditions, density=True)
    inverse_output, inverse_log_density = inference_network(
        forward_output, conditions=random_conditions, density=True, inverse=True
    )

    assert allclose(random_samples, inverse_output, atol=1e-3, rtol=1e-3)
    assert allclose(forward_log_density, inverse_log_density, atol=1e-3, rtol=1e-3)


@pytest.mark.torch
def test_density_numerically(inference_network, random_samples, random_conditions):
    import torch

    forward_output, forward_log_density = inference_network(random_samples, conditions=random_conditions, density=True)

    def f(x):
        return inference_network(x, conditions=random_conditions)

    numerical_forward_jacobian, *_ = torch.autograd.functional.jacobian(f, random_samples, vectorize=True)

    # TODO: torch is somehow permuted wrt keras
    numerical_forward_log_det = [
        keras.ops.log(keras.ops.abs(keras.ops.det(numerical_forward_jacobian[:, i, :])))
        for i in range(keras.ops.shape(random_samples)[0])
    ]
    numerical_forward_log_det = keras.ops.stack(numerical_forward_log_det, axis=0)

    log_prob = inference_network.base_distribution.log_prob(forward_output)

    numerical_forward_log_density = log_prob + numerical_forward_log_det

    assert allclose(forward_log_density, numerical_forward_log_density, rtol=1e-4, atol=1e-5)

    inverse_output, inverse_log_density = inference_network(
        random_samples, conditions=random_conditions, density=True, inverse=True
    )

    def f(x):
        return inference_network(x, conditions=random_conditions, inverse=True)

    numerical_inverse_jacobian, *_ = torch.autograd.functional.jacobian(f, random_samples, vectorize=True)

    # TODO: torch is somehow permuted wrt keras
    numerical_inverse_log_det = [
        keras.ops.log(keras.ops.abs(keras.ops.det(numerical_inverse_jacobian[:, i, :])))
        for i in range(keras.ops.shape(random_samples)[0])
    ]
    numerical_inverse_log_det = keras.ops.stack(numerical_inverse_log_det, axis=0)

    log_prob = inference_network.base_distribution.log_prob(random_samples)

    numerical_inverse_log_density = log_prob - numerical_inverse_log_det

    assert allclose(inverse_log_density, numerical_inverse_log_density, rtol=1e-4, atol=1e-5)


def test_serialize_deserialize(inference_network_subnet, subnet, random_samples, random_conditions):
    # to save, the model must be built
    inference_network_subnet(random_samples, conditions=random_conditions)

    serialized = serialize(inference_network_subnet)
    deserialized = deserialize(serialized)
    reserialized = serialize(deserialized)

    assert serialized == reserialized


def test_save_and_load(tmp_path, inference_network_subnet, subnet, random_samples, random_conditions):
    # to save, the model must be built
    inference_network_subnet(random_samples, conditions=random_conditions)

    keras.saving.save_model(inference_network_subnet, tmp_path / "model.keras")
    loaded = keras.saving.load_model(tmp_path / "model.keras")

    assert_layers_equal(inference_network_subnet, loaded)
