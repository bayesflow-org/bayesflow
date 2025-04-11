def test_scheduled_integration():
    import keras
    from bayesflow.utils import integrate

    def fn(t, x):
        return {"x": t**2}

    steps = keras.ops.convert_to_tensor([0.0, 0.5, 1.0])
    approximate_result = 0.0 + 0.5**2 * 0.5
    result = integrate(fn, {"x": 0.0}, steps=steps)["x"]
    assert result == approximate_result
