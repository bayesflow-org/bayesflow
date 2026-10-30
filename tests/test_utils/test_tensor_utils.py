import keras
import numpy as np
import pytest

from bayesflow._backend import grad
from bayesflow.utils import log_abs_det
from tests.utils import assert_allclose


pytestmark = pytest.mark.skip_on_mps


def test_log_abs_det():
    matrices = np.array(
        [
            [[2.0, 0.0], [0.0, 3.0]],
            [[1.0, 2.0], [3.0, 5.0]],
        ],
        dtype=np.float32,
    )

    actual = log_abs_det(keras.ops.convert_to_tensor(matrices))
    expected = np.linalg.slogdet(matrices)[1]

    assert_allclose(actual, expected, atol=1e-6)


def test_log_abs_det_is_differentiable():
    matrix = np.array([[2.0, 0.5], [0.25, 1.5]], dtype=np.float32)
    matrix_tensor = keras.ops.convert_to_tensor(matrix)

    actual = grad(log_abs_det)(matrix_tensor)
    expected = np.linalg.inv(matrix).T

    assert_allclose(actual, expected)


@pytest.mark.parametrize("elements_shape", [(4,), (4, 3)])
@pytest.mark.parametrize("weights_shape", [(4,), (4, 1)])
def test_weighted_mean_aligns_weights_with_batch_axis(elements_shape, weights_shape):
    from bayesflow.utils import weighted_mean

    elements = keras.random.normal(elements_shape)
    weights = keras.ops.reshape(keras.ops.arange(4, dtype="float32"), weights_shape)

    per_example = np.reshape(keras.ops.convert_to_numpy(elements), (4, -1)).mean(axis=-1)
    expected = np.mean(np.arange(4) * per_example)
    assert_allclose(weighted_mean(elements, weights), expected, rtol=1e-5, atol=1e-6)
