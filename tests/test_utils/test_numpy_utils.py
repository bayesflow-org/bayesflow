import keras
import numpy as np
import pytest

from bayesflow.utils import keras_utils, numpy_utils


@pytest.mark.parametrize("beta", [0.5, 1.0, 2.0])
@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_inverse_softplus_round_trip(beta, dtype):
    x = np.array([-20.0, -2.0, -0.5, 0.0, 0.5, 2.0, 19.0, 20.0, 21.0, 1000.0], dtype=dtype) / beta
    y = np.array([1e-8, 0.1, 1.0, 10.0, 30.0, 1000.0], dtype=dtype) / beta
    tolerance = 2e-6 if dtype == "float32" else 1e-8

    # The large values overflow the unused exponential branch, which must stay silent.
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        recovered_x = numpy_utils.inverse_softplus(numpy_utils.softplus(x, beta=beta), beta=beta)
        recovered_y = numpy_utils.softplus(numpy_utils.inverse_softplus(y, beta=beta), beta=beta)

    np.testing.assert_allclose(recovered_x, x, rtol=tolerance, atol=tolerance)
    np.testing.assert_allclose(recovered_y, y, rtol=tolerance, atol=0)


@pytest.mark.parametrize("beta", [0.5, 1.0, 2.0])
@pytest.mark.parametrize("shift", [np.log(np.e - 1), -1.5])
def test_inverse_shifted_softplus_round_trip(beta, shift):
    x = np.array([-2.0, -0.5, 0.0, 0.5, 2.0])
    y = numpy_utils.shifted_softplus(x, beta=beta, shift=shift)
    recovered = numpy_utils.inverse_shifted_softplus(y, beta=beta, shift=shift)

    np.testing.assert_allclose(recovered, x, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("beta", [0.5, 1.0, 2.0])
@pytest.mark.parametrize("threshold", [2.0, 20.0])
def test_inverse_softplus_matches_keras(beta, threshold):
    x = np.array([1e-6, 0.5, threshold - 0.25, threshold, threshold + 0.25, 1000.0], dtype="float32") / beta

    actual = numpy_utils.inverse_softplus(x, beta=beta, threshold=threshold)
    expected = keras.ops.convert_to_numpy(
        keras_utils.inverse_softplus(keras.ops.convert_to_tensor(x), beta=beta, threshold=threshold)
    )

    np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-6)


@pytest.mark.parametrize("beta", [0.5, 1.0, 2.0])
def test_inverse_softplus_threshold(beta):
    # A low threshold makes the strict boundary observable instead of rounding to x.
    threshold = 2.0
    x = np.array([threshold - 0.25, threshold, threshold + 0.25, 1000.0]) / beta

    with np.errstate(over="raise", invalid="raise", divide="raise"):
        actual = numpy_utils.inverse_softplus(x, beta=beta, threshold=threshold)

    # log(exp(z) - 1) = z + log(1 - exp(-z)), an independent stable reference.
    expected_below = x[:2] + np.log1p(-np.exp(-beta * x[:2])) / beta
    np.testing.assert_allclose(actual[:2], expected_below, rtol=1e-12, atol=1e-12)
    np.testing.assert_array_equal(actual[2:], x[2:])
