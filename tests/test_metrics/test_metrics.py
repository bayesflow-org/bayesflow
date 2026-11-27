from bayesflow.metrics import MaximumMeanDiscrepancy
from bayesflow.metrics.functional import maximum_mean_discrepancy
from bayesflow.metrics.functional.kernels import gaussian, inverse_multiquadratic
from bayesflow.utils.serialization import serialize, deserialize
import keras
import numpy as np
import pytest


def test_serialize_deserialize(metric):
    metric.update_state(keras.random.normal((2, 3)), keras.random.normal((2, 3)))

    serialized = serialize(metric)
    deserialized = deserialize(serialized)
    reserialized = serialize(deserialized)

    assert reserialized == serialized


@pytest.mark.parametrize(
    "kernel,kernel_fn",
    [
        ("inverse_multiquadratic", inverse_multiquadratic),
        pytest.param("gaussian", gaussian, marks=pytest.mark.skip_on_mps),
    ],
)
def test_unbiased_mmd_matches_gretton_formula(kernel, kernel_fn):
    rng = np.random.default_rng(0)
    x = keras.ops.convert_to_tensor(rng.standard_normal((12, 3)), dtype="float32")
    y = keras.ops.convert_to_tensor(rng.standard_normal((17, 3)) + 0.5, dtype="float32")

    actual = float(keras.ops.convert_to_numpy(maximum_mean_discrepancy(x, y, kernel=kernel, unbiased=True)))

    k_xx = keras.ops.convert_to_numpy(kernel_fn(x, x))
    k_yy = keras.ops.convert_to_numpy(kernel_fn(y, y))
    k_xy = keras.ops.convert_to_numpy(kernel_fn(x, y))
    m, n = len(k_xx), len(k_yy)
    expected = (
        (k_xx.sum() - np.trace(k_xx)) / (m * (m - 1))
        + (k_yy.sum() - np.trace(k_yy)) / (n * (n - 1))
        - 2.0 * k_xy.mean()
    )

    assert np.isclose(actual, expected, rtol=1e-5, atol=1e-5)

    metric = MaximumMeanDiscrepancy(kernel=kernel, unbiased=True)
    metric.update_state(x, y)
    assert np.isclose(float(keras.ops.convert_to_numpy(metric.result())), expected, rtol=1e-5, atol=1e-5)


def test_unbiased_mmd_zero_expectation_under_null():
    rng = np.random.default_rng(42)
    estimates = []
    for _ in range(100):
        x = keras.ops.convert_to_tensor(rng.standard_normal((25, 2)), dtype="float32")
        y = keras.ops.convert_to_tensor(rng.standard_normal((30, 2)), dtype="float32")
        estimates.append(float(keras.ops.convert_to_numpy(maximum_mean_discrepancy(x, y, unbiased=True))))

    mean_est = float(np.mean(estimates))
    se_est = float(np.std(estimates, ddof=1) / np.sqrt(len(estimates)))
    assert abs(mean_est) < 4.0 * se_est


def test_unbiased_mmd_requires_at_least_two_samples():
    x = keras.ops.convert_to_tensor(np.zeros((1, 3), dtype="float32"))
    y = keras.ops.convert_to_tensor(np.ones((4, 3), dtype="float32"))

    with pytest.raises(ValueError, match="at least 2 samples"):
        maximum_mean_discrepancy(x, y, unbiased=True)
    with pytest.raises(ValueError, match="at least 2 samples"):
        maximum_mean_discrepancy(y, x, unbiased=True)
