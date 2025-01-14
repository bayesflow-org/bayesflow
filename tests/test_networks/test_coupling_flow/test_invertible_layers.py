import keras
import numpy as np

from tests.utils import assert_allclose


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

    assert_allclose(random_samples, inverse_output, atol=1e-6, msg="Samples are not cycle consistent")
    assert_allclose(forward_log_det, -inverse_log_det, atol=1e-6, msg="Log Determinants are not cycle consistent")


def test_jacobian_numerically(invertible_layer, random_samples, random_conditions):
    from bayesflow.utils import jacobian

    forward_output, forward_log_det = invertible_layer(random_samples)

    numerical_forward_jacobian = jacobian(lambda x: invertible_layer(x)[0], random_samples)

    numerical_forward_log_det = keras.ops.logdet(numerical_forward_jacobian)

    assert_allclose(forward_log_det, numerical_forward_log_det, rtol=1e-4, atol=1e-5)

    inverse_output, inverse_log_det = invertible_layer(random_samples, inverse=True)

    numerical_inverse_jacobian = jacobian(lambda z: invertible_layer(z, inverse=True)[0], random_samples)

    numerical_inverse_log_det = keras.ops.logdet(numerical_inverse_jacobian)

    assert_allclose(inverse_log_det, numerical_inverse_log_det, rtol=1e-4, atol=1e-5)
