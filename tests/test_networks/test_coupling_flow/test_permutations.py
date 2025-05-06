import pytest
import keras
import numpy as np

from bayesflow.networks.coupling_flow.permutations import (
    FixedPermutation,
    OrthogonalPermutation,
    RandomPermutation,
    Swap,
)


@pytest.fixture(params=[FixedPermutation, OrthogonalPermutation, RandomPermutation, Swap])
def permutation_class(request):
    return request.param


@pytest.fixture
def input_tensor():
    return keras.random.normal((2, 5))


def test_fixed_permutation_build_and_call():
    # Since FixedPermutation is abstract, create a subclass for testing build.
    class TestPerm(FixedPermutation):
        def build(self, xz_shape, **kwargs):
            length = xz_shape[-1]
            self.forward_indices = keras.ops.arange(length - 1, -1, -1)
            self.inverse_indices = keras.ops.arange(length - 1, -1, -1)

    layer = TestPerm()
    input_shape = (2, 4)
    layer.build(input_shape)

    x = keras.ops.convert_to_tensor(np.arange(8).reshape(input_shape).astype("float32"))
    z, log_det = layer(x, inverse=False)
    x_inv, log_det_inv = layer(z, inverse=True)

    # Check shape preservation
    assert z.shape == x.shape
    assert x_inv.shape == x.shape
    # Forward then inverse recovers input
    np.testing.assert_allclose(keras.ops.convert_to_numpy(x_inv), keras.ops.convert_to_numpy(x), atol=1e-5)
    # log_det values should be zero tensors with the correct shape
    assert tuple(log_det.shape) == input_shape[:-1]
    assert tuple(log_det_inv.shape) == input_shape[:-1]


def test_orthogonal_permutation_build_and_call(input_tensor):
    layer = OrthogonalPermutation()
    input_shape = keras.ops.shape(input_tensor)
    layer.build(input_shape)

    z, log_det = layer(input_tensor)
    x_inv, log_det_inv = layer(z, inverse=True)

    # Check output shapes
    assert z.shape == input_tensor.shape
    assert x_inv.shape == input_tensor.shape

    # Forward + inverse should approximately recover input (allow some numeric tolerance)
    np.testing.assert_allclose(
        keras.ops.convert_to_numpy(x_inv), keras.ops.convert_to_numpy(input_tensor), rtol=1e-5, atol=1e-5
    )

    # log_det should be scalar or batched scalar
    if len(log_det.shape) > 0:
        assert log_det.shape[0] == input_tensor.shape[0]  # batch dim
    else:
        assert log_det.shape == ()

    # log_det_inv should be negative of log_det (det(inv) = 1/det)
    log_det_np = keras.ops.convert_to_numpy(log_det)
    log_det_inv_np = keras.ops.convert_to_numpy(log_det_inv)
    np.testing.assert_allclose(log_det_inv_np, -log_det_np, rtol=1e-5, atol=1e-5)


def test_random_permutation_build_and_call(input_tensor):
    layer = RandomPermutation()
    input_shape = keras.ops.shape(input_tensor)
    layer.build(input_shape)

    # Assert forward_indices and inverse_indices are set and consistent
    fwd = keras.ops.convert_to_numpy(layer.forward_indices)
    inv = keras.ops.convert_to_numpy(layer.inverse_indices)
    # Applying inv on fwd must yield ordered indices
    reordered = fwd[inv]
    np.testing.assert_array_equal(np.arange(len(fwd)), reordered)

    z, log_det = layer(input_tensor)
    x_inv, log_det_inv = layer(z, inverse=True)

    assert z.shape == input_tensor.shape
    assert x_inv.shape == input_tensor.shape
    np.testing.assert_allclose(keras.ops.convert_to_numpy(x_inv), keras.ops.convert_to_numpy(input_tensor), atol=1e-5)
    assert tuple(log_det.shape) == input_shape[:-1]
    assert tuple(log_det_inv.shape) == input_shape[:-1]


def test_swap_build_and_call(input_tensor):
    layer = Swap()
    input_shape = keras.ops.shape(input_tensor)
    layer.build(input_shape)

    fwd = keras.ops.convert_to_numpy(layer.forward_indices)
    inv = keras.ops.convert_to_numpy(layer.inverse_indices)
    reordered = fwd[inv]
    np.testing.assert_array_equal(np.arange(len(fwd)), reordered)

    z, log_det = layer(input_tensor)
    x_inv, log_det_inv = layer(z, inverse=True)

    assert z.shape == input_tensor.shape
    assert x_inv.shape == input_tensor.shape
    np.testing.assert_allclose(keras.ops.convert_to_numpy(x_inv), keras.ops.convert_to_numpy(input_tensor), atol=1e-5)
    assert tuple(log_det.shape) == input_shape[:-1]
    assert tuple(log_det_inv.shape) == input_shape[:-1]
