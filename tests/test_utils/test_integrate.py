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
    T_final = 1.0
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
    "method, atol", [("euler", TOLERANCE_EULER), ("rk45", TOLERANCE_ADAPTIVE), ("tsit5", TOLERANCE_ADAPTIVE)]
)
def test_analytical_backward_integration(method, atol):
    T_final = 1.0

    def fn(t, x):
        return {"x": keras.ops.convert_to_tensor([2.0 * t])}

    num_steps = 100
    analytical_result = 1.0
    initial_state = {"x": keras.ops.convert_to_tensor([1.0 + T_final**2])}

    result = integrate(fn, initial_state, start_time=T_final, stop_time=0.0, steps=num_steps, method=method)["x"]
    if method == "euler":
        result_adaptive = result
    else:
        result_adaptive = integrate(
            fn, initial_state, start_time=T_final, stop_time=0.0, steps="adaptive", method=method, max_steps=1_000
        )["x"]

    np.testing.assert_allclose(result, analytical_result, atol=atol, rtol=0.1)
    np.testing.assert_allclose(result_adaptive, analytical_result, atol=atol, rtol=0.1)


@pytest.mark.parametrize(
    "method,use_adapt",
    [
        ("euler_maruyama", False),
        ("euler_maruyama", True),
        ("sea", False),
        ("shark", False),
        ("fast_adaptive", False),
        ("fast_adaptive", True),
    ],
)
def test_forward_additive_ou_weak_means_and_vars(method, use_adapt):
    """
    Ornstein-Uhlenbeck with additive noise, integrated FORWARD in time.
    This serves as a sanity check that forward integration still works correctly.

    Forward SDE:
        dX = a X dt + sigma dW

    Exact at time T starting from X(0) = x_0:
        E[X(T)] = x_0 * exp(a T)
        Var[X(T)] = sigma^2 * (exp(2 a T) - 1) / (2 a)
    """
    # SDE parameters
    a = -1.0
    sigma = 0.5
    x_0 = 1.2  # initial condition at time 0
    T = 1.0

    # batch of trajectories
    N = 10000
    seed = keras.random.SeedGenerator(42)

    def drift_fn(t, x):
        return {"x": a * x}

    def diffusion_fn(t, x):
        return {"x": keras.ops.convert_to_tensor([sigma])}

    initial_state = {"x": keras.ops.ones((N,)) * x_0}
    steps = 200 if not use_adapt else "adaptive"

    # Expected mean and variance at t=T
    exp_mean = x_0 * np.exp(a * T)
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

    x_T = np.array(out["x"])
    emp_mean = float(x_T.mean())
    emp_var = float(x_T.var())

    np.testing.assert_allclose(emp_mean, exp_mean, atol=TOL_MEAN, rtol=0.0)
    np.testing.assert_allclose(emp_var, exp_var, atol=TOL_VAR, rtol=0.0)


@pytest.mark.parametrize(
    "method,use_adapt",
    [
        ("euler_maruyama", False),
        ("euler_maruyama", True),
        ("sea", False),
        ("shark", False),
        ("fast_adaptive", False),
        ("fast_adaptive", True),
    ],
)
def test_backward_additive_ou_weak_means_and_vars(method, use_adapt):
    """
    Ornstein-Uhlenbeck with additive noise, integrated BACKWARD in time.

    When integrating from t=T back to t=0 with initial condition X(T) = x_T,
    we get X(0) which should satisfy:
        E[X(0)] = x_T * exp(-a T)  (-a because we go backward)
        Var[X(0)] = sigma^2 * (exp(-2 a T) - 1) / (-2 a)

    We verify weak accuracy by matching empirical mean and variance.
    """
    # SDE parameters
    a = -1.0
    sigma = 0.5
    x_T = 1.2  # initial condition at time T
    T = 1.0

    # batch of trajectories
    N = 10000  # large enough to control sampling error
    seed = keras.random.SeedGenerator(42)

    def drift_fn(t, x):
        return {"x": a * x}

    def diffusion_fn(t, x):
        # additive noise, independent of state
        return {"x": keras.ops.convert_to_tensor([sigma])}

    # Start at time T with value x_T
    initial_state = {"x": keras.ops.ones((N,)) * x_T}
    steps = 200 if not use_adapt else "adaptive"

    # Expected mean and variance at t=0 after integrating backward from t=T
    # For backward integration, the effective drift coefficient changes sign
    exp_mean = x_T * np.exp(-a * T)
    exp_var = sigma**2 * (np.exp(-2.0 * a * T) - 1.0) / (-2.0 * a)

    out = integrate_stochastic(
        drift_fn=drift_fn,
        diffusion_fn=diffusion_fn,
        state=initial_state,
        start_time=T,
        stop_time=0.0,
        steps=steps,
        seed=seed,
        method=method,
        max_steps=1_000,
    )

    x_0 = np.array(out["x"])
    emp_mean = float(x_0.mean())
    emp_var = float(x_0.var())

    np.testing.assert_allclose(emp_mean, exp_mean, atol=TOL_MEAN, rtol=0.0)
    np.testing.assert_allclose(emp_var, exp_var, atol=TOL_VAR, rtol=0.0)


@pytest.mark.parametrize(
    "method,use_adapt",
    [
        ("euler_maruyama", False),
        ("euler_maruyama", True),
        ("sea", False),
        ("shark", False),
        ("fast_adaptive", False),
        ("fast_adaptive", True),
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
    seed = keras.random.SeedGenerator(0)

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


@pytest.mark.parametrize("steps", [500])
def test_langevin_gaussian_sampling(steps):
    """
    Test annealed Langevin dynamics on a 1D Gaussian target.

    Target distribution: N(mu, sigma^2), with score
        ∇_x log p(x) = -(x - mu) / sigma^2

    We verify that the empirical mean and variance after Langevin sampling
    match the target within a loose tolerance (to allow for Monte Carlo noise).
    """
    # target parameters
    mu = 0.3
    sigma = 0.7

    # number of particles
    N = 20000
    start_time = 0.0
    stop_time = 1.0

    # tolerances for mean and variance
    tol_mean = 5e-2
    tol_var = 5e-2

    # initial state: broad Gaussian, independent of target
    seed = keras.random.SeedGenerator(42)
    x0 = keras.random.normal((N,), dtype="float32", seed=seed)
    initial_state = {"x": x0}

    # simple dummy noise schedule: constant alpha
    class DummyNoiseSchedule:
        def get_log_snr(self, t, training=False):
            return keras.ops.zeros_like(t)

        def get_alpha_sigma(self, log_snr_t):
            alpha_t = keras.ops.ones_like(log_snr_t)
            sigma_t = keras.ops.ones_like(log_snr_t)
            return alpha_t, sigma_t

    noise_schedule = DummyNoiseSchedule()

    # score of the target Gaussian
    def score_fn(t, x):
        s = -(x - mu) / (sigma**2)
        return {"x": s}

    # run Langevin
    final_state = integrate_stochastic(
        drift_fn=None,
        diffusion_fn=None,
        score_fn=score_fn,
        noise_schedule=noise_schedule,
        state=initial_state,
        start_time=start_time,
        stop_time=stop_time,
        steps=steps,
        seed=seed,
        method="langevin",
        max_steps=1_000,
        corrector_steps=1,
    )

    xT = np.array(final_state["x"])
    emp_mean = float(xT.mean())
    emp_var = float(xT.var())

    exp_mean = mu
    exp_var = sigma**2

    np.testing.assert_allclose(emp_mean, exp_mean, atol=tol_mean, rtol=0.0)
    np.testing.assert_allclose(emp_var, exp_var, atol=tol_var, rtol=0.0)
