import numpy as np
import keras
import pytest
from bayesflow.utils import integrate


TOLERANCE_ADAPTIVE = 1e-6  # Adaptive solvers should be very accurate.
TOLERANCE_EULER = 1e-3  # Euler with fixed steps requires a larger tolerance


def test_scheduled_integration():
    import keras
    from bayesflow.utils import integrate

    def fn(t, x):
        return {"x": t**2}

    steps = keras.ops.convert_to_tensor([0.0, 0.5, 1.0])
    approximate_result = 0.0 + 0.5**2 * 0.5
    result = integrate(fn, {"x": 0.0}, steps=steps)["x"]
    assert result == approximate_result


def test_scipy_integration():
    import keras
    from bayesflow.utils import integrate

    def fn(t, x):
        return {"x": keras.ops.exp(t)}

    start_time = -1.0
    stop_time = 1.0
    exact_result = keras.ops.exp(stop_time) - keras.ops.exp(start_time)
    result = integrate(
        fn,
        {"x": 0.0},
        start_time=start_time,
        stop_time=stop_time,
        steps="adaptive",
        method="scipy",
        scipy_kwargs={"atol": 1e-6, "rtol": 1e-6},
    )["x"]
    np.testing.assert_allclose(exact_result, result, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize(
    "method, atol", [("euler", TOLERANCE_EULER), ("rk45", TOLERANCE_ADAPTIVE), ("tsit5", TOLERANCE_ADAPTIVE)]
)
def test_analytical_integration(method, atol):
    def fn(t, x):
        return {"x": keras.ops.convert_to_tensor([2.0 * t])}

    initial_state = {"x": keras.ops.convert_to_tensor([1.0])}
    T_final = 2.0
    num_steps = 100
    analytical_result = 1.0 + T_final**2

    result = integrate(fn, initial_state, start_time=0.0, stop_time=T_final, steps=num_steps, method=method)["x"]
    result_adaptive = integrate(
        fn, initial_state, start_time=0.0, stop_time=T_final, steps="adaptive", method=method, max_steps=1_000
    )["x"]
    np.testing.assert_allclose(result, analytical_result, atol=atol, rtol=0.1)

    np.testing.assert_allclose(result_adaptive, analytical_result, atol=atol, rtol=0.01)
