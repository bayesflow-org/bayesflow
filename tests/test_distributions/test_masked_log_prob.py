import keras
import pytest

from tests.utils import assert_allclose


def test_mask_of_ones_is_moot(diagonal_normal, random_samples):
    diagonal_normal.build(keras.ops.shape(random_samples))
    ones = keras.ops.ones_like(random_samples)

    assert_allclose(diagonal_normal.log_prob(random_samples, mask=ones), diagonal_normal.log_prob(random_samples))


def test_student_t_raises(diagonal_student_t, random_samples):
    """Its dims share one chi-square draw, so dropping dims does not give the marginal."""
    diagonal_student_t.build(keras.ops.shape(random_samples))

    with pytest.raises(NotImplementedError, match="multivariate t"):
        diagonal_student_t.log_prob(random_samples, mask=keras.ops.ones_like(random_samples))
