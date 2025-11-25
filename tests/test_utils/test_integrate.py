import numpy as np
import keras
import pytest
from bayesflow.utils import integrate, integrate_stochastic


TOLERANCE_ADAPTIVE = 1e-6  # Adaptive solvers should be very accurate.
TOLERANCE_EULER = 1e-3  # Euler with fixed steps requires a larger tolerance

# tolerances for SDE tests
TOL_MEAN = 3e-2
TOL_VAR = 5e-2
TOL_DET = 1e-3


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
    if method == "euler":
        result_adaptive = result
    else:
        result_adaptive = integrate(
            fn, initial_state, start_time=0.0, stop_time=T_final, steps="adaptive", method=method, max_steps=1_000
        )["x"]

    np.testing.assert_allclose(result, analytical_result, atol=atol, rtol=0.1)
    np.testing.assert_allclose(result_adaptive, analytical_result, atol=atol, rtol=0.1)


@pytest.mark.parametrize(
    "method,use_adapt",
    [
        ("euler_maruyama", False),
        ("shark", False),
        ("shark", True),
    ],
)
def test_additive_OU_weak_means_and_vars(method, use_adapt):
    """
    Ornstein Uhlenbeck with additive noise
        dX = a X dt + sigma dW
    Exact at time T:
        E[X_T] = x0 * exp(a T)
        Var[X_T] = sigma^2 * (exp(2 a T) - 1) / (2 a)
    We verify weak accuracy by matching empirical mean and variance.
    """
    # SDE parameters
    a = -1.0
    sigma = 0.5
    x0 = 1.2
    T = 1.0

    # batch of trajectories
    N = 20000  # large enough to control sampling error
    seed = keras.random.SeedGenerator(42)

    def drift_fn(t, x):
        return {"x": a * x}

    def diffusion_fn(t, x):
        # additive noise, independent of state
        return {"x": keras.ops.convert_to_tensor([sigma])}

    initial_state = {"x": keras.ops.ones((N,)) * x0}
    steps = 200 if not use_adapt else "adaptive"

    # expected mean and variance
    exp_mean = x0 * np.exp(a * T)
    exp_var = sigma**2 * (np.exp(2.0 * a * T) - 1.0) / (2.0 * a)

    out = integrate_stochastic(
        drift_fn=drift_fn,
        diffusion_fn=diffusion_fn,
        state=initial_state,
        start_time=0.0,
        stop_time=T,
        steps=steps,
        seed=seed,
        method=method,
        max_steps=1_000,
    )

    xT = np.array(out["x"])
    emp_mean = float(xT.mean())
    emp_var = float(xT.var())
    np.testing.assert_allclose(emp_mean, exp_mean, atol=TOL_MEAN, rtol=0.0)
    np.testing.assert_allclose(emp_var, exp_var, atol=TOL_VAR, rtol=0.0)


@pytest.mark.parametrize(
    "method,use_adapt",
    [
        ("euler_maruyama", False),
        ("shark", False),
        ("shark", True),
    ],
)
def test_zero_noise_reduces_to_deterministic(method, use_adapt):
    """
    With zero diffusion the SDE reduces to the ODE
        dX = a X dt
    """
    a = 0.7
    x0 = 0.9
    T = 1.25
    steps = 200 if not use_adapt else "adaptive"
    seed = keras.random.SeedGenerator(999)

    def drift_fn(t, x):
        return {"x": a * x}

    def diffusion_fn(t, x):
        # identically zero diffusion
        return {"x": keras.ops.convert_to_tensor([0.0])}

    initial_state = {"x": keras.ops.ones((256,)) * x0}
    out = integrate_stochastic(
        drift_fn=drift_fn,
        diffusion_fn=diffusion_fn,
        state=initial_state,
        start_time=0.0,
        stop_time=T,
        steps=steps,
        seed=seed,
        method=method,
        max_steps=1_000,
    )["x"]

    exact = x0 * np.exp(a * T)
    np.testing.assert_allclose(np.array(out).mean(), exact, atol=TOL_DET, rtol=0.1)
