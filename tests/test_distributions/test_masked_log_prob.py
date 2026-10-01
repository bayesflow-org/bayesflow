import keras
import numpy as np
import pytest

from tests.utils import assert_allclose


def test_mask_of_ones_is_moot(diagonal_normal, random_samples):
    diagonal_normal.build(keras.ops.shape(random_samples))
    ones = keras.ops.ones_like(random_samples)

    assert_allclose(diagonal_normal.log_prob(random_samples, mask=ones), diagonal_normal.log_prob(random_samples))


@pytest.mark.parametrize("normalize", [True, False])
def test_masked_normal_matches_lower_dim_normal(diagonal_normal, random_samples, normalize):
    """Dropping the last dim must give the density of a normal over the remaining dims."""
    from bayesflow.distributions import DiagonalNormal

    diagonal_normal.build(keras.ops.shape(random_samples))
    mask = np.ones(keras.ops.shape(random_samples), dtype="float32")
    mask[..., -1] = 0
    kept = random_samples[..., :-1]
    reference = DiagonalNormal()
    reference.build(keras.ops.shape(kept))

    assert_allclose(
        diagonal_normal.log_prob(random_samples, normalize=normalize, mask=mask),
        reference.log_prob(kept, normalize=normalize),
    )


def test_student_t_raises(diagonal_student_t, random_samples):
    """Its dims share one chi-square draw, so dropping dims does not give the marginal."""
    diagonal_student_t.build(keras.ops.shape(random_samples))

    with pytest.raises(NotImplementedError, match="multivariate t"):
        diagonal_student_t.log_prob(random_samples, mask=keras.ops.ones_like(random_samples))
