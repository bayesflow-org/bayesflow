import numpy as np


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
