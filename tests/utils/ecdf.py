import numpy as np
import pytest
from bayesflow.utils.ecdf import fractional_ranks, distance_ranks, random_ranks


@pytest.fixture
def test_data():
    # Provide sample data for testing
    np.random.seed(42)  # Set seed for reproducibility
    post_samples = np.array([[[0.1, 0.2], [0.3, 0.4]], [[0.5, 0.6], [0.7, 0.8]]])
    prior_samples = np.array([[0.2, 0.3], [0.6, 0.7]])
    references = np.array([[0.0, 0.0], [0.0, 0.0]])
    return post_samples, prior_samples, references


def test_fractional_ranks(test_data):
    post_samples, prior_samples, _ = test_data
    # Compute expected result manually
    expected = np.mean(post_samples < prior_samples[:, np.newaxis, :], axis=1)
    result = fractional_ranks(post_samples, prior_samples)
    np.testing.assert_almost_equal(result, expected, decimal=6)


@pytest.mark.parametrize("stacked, expected_shape", [(True, (2, 1)), (False, (2, 2))])
def test_distance_ranks(test_data, stacked, expected_shape):
    post_samples, prior_samples, _ = test_data
    result = distance_ranks(post_samples=post_samples, prior_samples=prior_samples, stacked=stacked)
    assert result.shape == expected_shape


@pytest.mark.parametrize("stacked, expected_shape", [(True, (2, 1)), (False, (2, 2))])
def test_random_ranks(test_data, stacked, expected_shape):
    post_samples, prior_samples, _ = test_data
    result = random_ranks(post_samples=post_samples, prior_samples=prior_samples, stacked=stacked)
    assert result.shape == expected_shape
