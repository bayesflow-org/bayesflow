import functools

import keras
import numpy as np
import pytest

from tests.utils import allclose


def test_build(invertible_layer, random_samples, random_conditions):
    assert invertible_layer.built is False

    invertible_layer(random_samples)

    assert invertible_layer.built is True

    assert invertible_layer.variables, "Layer has no variables."


def test_variable_batch_size(invertible_layer, random_samples, random_conditions):
    # manual build with one batch size
    invertible_layer.build(keras.ops.shape(random_samples))

    # run with another batch size
    batch_sizes = np.random.choice(10, replace=False, size=3)
    for batch_size in batch_sizes:
        new_input = keras.ops.zeros((batch_size,) + keras.ops.shape(random_samples)[1:])
        invertible_layer(new_input)


def test_output_structure(invertible_layer, random_samples, random_conditions):
    output = invertible_layer(random_samples)

    assert isinstance(output, tuple)
    assert len(output) == 2

    forward_output, forward_log_det = output

    assert keras.ops.is_tensor(forward_output)
    assert keras.ops.is_tensor(forward_log_det)


def test_output_shape(invertible_layer, random_samples, random_conditions):
    forward_output, forward_log_det = invertible_layer(random_samples)

    assert keras.ops.shape(forward_output) == keras.ops.shape(random_samples)
    assert keras.ops.shape(forward_log_det) == (keras.ops.shape(random_samples)[0],)

    inverse_output, inverse_log_det = invertible_layer(random_samples, inverse=True)

    assert keras.ops.shape(inverse_output) == keras.ops.shape(random_samples)
    assert keras.ops.shape(inverse_log_det) == (keras.ops.shape(random_samples)[0],)


def test_cycle_consistency(invertible_layer, random_samples, random_conditions):
    # cycle-consistency means the forward and inverse methods are inverses of each other
    forward_output, forward_log_det = invertible_layer(random_samples)
    inverse_output, inverse_log_det = invertible_layer(forward_output, inverse=True)

    assert allclose(random_samples, inverse_output)
    assert allclose(forward_log_det, -inverse_log_det)


@pytest.mark.torch
def test_jacobian_numerically(invertible_layer, random_samples, random_conditions):
    import torch

    forward_output, forward_log_det = invertible_layer(random_samples)
    numerical_forward_jacobian, *_ = torch.autograd.functional.jacobian(
        invertible_layer, random_samples, vectorize=True
    )

    # TODO: torch is somehow permuted wrt keras
    numerical_forward_log_det = [
        keras.ops.log(keras.ops.abs(keras.ops.det(numerical_forward_jacobian[i, :, i, :])))
        for i in range(keras.ops.shape(random_samples)[0])
    ]
    numerical_forward_log_det = keras.ops.stack(numerical_forward_log_det, axis=0)

    assert allclose(forward_log_det, numerical_forward_log_det, rtol=1e-4, atol=1e-5)

    inverse_output, inverse_log_det = invertible_layer(random_samples, inverse=True)

    numerical_inverse_jacobian, *_ = torch.autograd.functional.jacobian(
        functools.partial(invertible_layer, inverse=True), random_samples, vectorize=True
    )

    # TODO: torch is somehow permuted wrt keras
    numerical_inverse_log_det = [
        keras.ops.log(keras.ops.abs(keras.ops.det(numerical_inverse_jacobian[i, :, i, :])))
        for i in range(keras.ops.shape(random_samples)[0])
    ]
    numerical_inverse_log_det = keras.ops.stack(numerical_inverse_log_det, axis=0)

    assert allclose(inverse_log_det, numerical_inverse_log_det, rtol=1e-4, atol=1e-5)
