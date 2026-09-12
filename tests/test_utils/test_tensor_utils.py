import keras
import numpy as np

from bayesflow._backend import grad
from bayesflow.utils import log_abs_det
from tests.utils import assert_allclose


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
