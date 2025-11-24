from collections.abc import Callable, Sequence
from functools import partial

import keras

import numpy as np
from typing import Literal, Union

from bayesflow.adapters import Adapter
from bayesflow.types import Tensor
from bayesflow.utils import filter_kwargs
from bayesflow.utils.logging import warning

from . import logging

ArrayLike = int | float | Tensor


def euler_step(
    fn: Callable,
    state: dict[str, ArrayLike],
    time: ArrayLike,
    step_size: ArrayLike,
    tolerance: ArrayLike = 1e-6,
    min_step_size: ArrayLike = -float("inf"),
    max_step_size: ArrayLike = float("inf"),
    use_adaptive_step_size: bool = False,
) -> (dict[str, ArrayLike], ArrayLike, ArrayLike):
    k1 = fn(time, **filter_kwargs(state, fn))

    if use_adaptive_step_size:
        # Use Heun's method (RK2) as embedded pair for proper error estimation
        intermediate_state = state.copy()
        for key, delta in k1.items():
            intermediate_state[key] = state[key] + step_size * delta

        k2 = fn(time + step_size, **filter_kwargs(intermediate_state, fn))

        # check all keys are equal
        if set(k1.keys()) != set(k2.keys()):
            raise ValueError("Keys of the deltas do not match. Please return zero for unchanged variables.")

        # Heun's (RK2) solution
        heun_state = state.copy()
        for key in k1.keys():
            heun_state[key] = state[key] + 0.5 * step_size * (k1[key] + k2[key])

        # Error estimate: difference between Euler and Heun
        intermediate_error = keras.ops.stack(
            [keras.ops.norm(heun_state[key] - intermediate_state[key], ord=2, axis=-1) for key in k1]
        )

        max_error = keras.ops.max(intermediate_error)
        new_step_size = step_size * keras.ops.sqrt(tolerance / (max_error + 1e-9))

        new_step_size = keras.ops.clip(new_step_size, min_step_size, max_step_size)
    else:
        new_step_size = step_size

    new_state = state.copy()
    for key in k1.keys():
        new_state[key] = state[key] + step_size * k1[key]

    new_time = time + step_size

    return new_state, new_time, new_step_size


def add_scaled(state, ks, coeffs, h):
    out = {}
    for key, y in state.items():
        acc = keras.ops.zeros_like(y)
        for c, k in zip(coeffs, ks):
            acc = acc + c * k[key]
        out[key] = y + h * acc
    return out


def rk45_step(
    fn: Callable,
    state: dict[str, ArrayLike],
    time: ArrayLike,
    last_step_size: ArrayLike,
    tolerance: ArrayLike = 1e-6,
    min_step_size: ArrayLike = -float("inf"),
    max_step_size: ArrayLike = float("inf"),
    use_adaptive_step_size: bool = False,
) -> (dict[str, ArrayLike], ArrayLike, ArrayLike):
    """
    Dormand-Prince 5(4) method with embedded error estimation.
    """
    step_size = last_step_size
    h = step_size

    k1 = fn(time, **filter_kwargs(state, fn))
    k2 = fn(time + h * (1 / 5), **add_scaled(state, [k1], [1 / 5], h))
    k3 = fn(time + h * (3 / 10), **add_scaled(state, [k1, k2], [3 / 40, 9 / 40], h))
    k4 = fn(time + h * (4 / 5), **add_scaled(state, [k1, k2, k3], [44 / 45, -56 / 15, 32 / 9], h))
    k5 = fn(
        time + h * (8 / 9),
        **add_scaled(state, [k1, k2, k3, k4], [19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729], h),
    )
    k6 = fn(
        time + h,
        **add_scaled(state, [k1, k2, k3, k4, k5], [9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656], h),
    )

    # check all keys are equal
    if not all(set(k.keys()) == set(k1.keys()) for k in [k2, k3, k4, k5, k6]):
        raise ValueError("Keys of the deltas do not match. Please return zero for unchanged variables.")

    # 5th order solution
    new_state = {}
    for key in k1.keys():
        new_state[key] = state[key] + h * (
            35 / 384 * k1[key] + 500 / 1113 * k3[key] + 125 / 192 * k4[key] - 2187 / 6784 * k5[key] + 11 / 84 * k6[key]
        )

    if use_adaptive_step_size:
        k7 = fn(time + h, **filter_kwargs(new_state, fn))

        # 4th order embedded solution
        err_state = {}
        for key in k1.keys():
            y4 = state[key] + h * (
                5179 / 57600 * k1[key]
                + 7571 / 16695 * k3[key]
                + 393 / 640 * k4[key]
                - 92097 / 339200 * k5[key]
                + 187 / 2100 * k6[key]
                + 1 / 40 * k7[key]
            )
            err_state[key] = new_state[key] - y4

        err_norm = keras.ops.stack([keras.ops.norm(v, ord=2, axis=-1) for v in err_state.values()])
        err = keras.ops.max(err_norm)

        new_step_size = h * keras.ops.clip(0.9 * (tolerance / (err + 1e-12)) ** 0.2, 0.2, 5.0)
        new_step_size = keras.ops.clip(new_step_size, min_step_size, max_step_size)
    else:
        new_step_size = step_size

    new_time = time + h
    return new_state, new_time, new_step_size


def tsit5_step(
    fn: Callable,
    state: dict[str, ArrayLike],
    time: ArrayLike,
    last_step_size: ArrayLike,
    tolerance: ArrayLike = 1e-6,
    min_step_size: ArrayLike = -float("inf"),
    max_step_size: ArrayLike = float("inf"),
    use_adaptive_step_size: bool = False,
):
    """
    Implements a single step of the Tsitouras 5/4 Runge-Kutta method.
    """
    step_size = last_step_size
    h = step_size

    # Butcher tableau coefficients
    c2 = 0.161
    c3 = 0.327
    c4 = 0.9
    c5 = 0.9800255409045097

    k1 = fn(time, **filter_kwargs(state, fn))
    k2 = fn(time + h * c2, **add_scaled(state, [k1], [0.161], h))
    k3 = fn(time + h * c3, **add_scaled(state, [k1, k2], [-0.0084806554923570, 0.3354806554923570], h))
    k4 = fn(
        time + h * c4, **add_scaled(state, [k1, k2, k3], [2.897153057105494, -6.359448489975075, 4.362295432869581], h)
    )
    k5 = fn(
        time + h * c5,
        **add_scaled(
            state, [k1, k2, k3, k4], [4.325279681768730, -11.74888356406283, 7.495539342889836, -0.09249506636175525], h
        ),
    )
    k6 = fn(
        time + h,
        **add_scaled(
            state,
            [k1, k2, k3, k4, k5],
            [5.86145544294270, -12.92096931784711, 8.159367898576159, -0.07158497328140100, -0.02826905039406838],
            h,
        ),
    )

    # 5th order solution: b coefficients
    new_state = {}
    for key in state.keys():
        new_state[key] = state[key] + h * (
            0.09646076681806523 * k1[key]
            + 0.01 * k2[key]
            + 0.4798896504144996 * k3[key]
            + 1.379008574103742 * k4[key]
            - 3.290069515436081 * k5[key]
            + 2.324710524099774 * k6[key]
        )

    if use_adaptive_step_size:
        # 7th stage evaluation
        k7 = fn(time + h, **filter_kwargs(new_state, fn))

        # 4th order embedded solution: b_hat coefficients
        y4 = {}
        for key in state.keys():
            y4[key] = state[key] + h * (
                0.001780011052226 * k1[key]
                + 0.000816434459657 * k2[key]
                - 0.007880878010262 * k3[key]
                + 0.144711007173263 * k4[key]
                - 0.582357165452555 * k5[key]
                + 0.458082105929187 * k6[key]
                + (1.0 / 66.0) * k7[key]
            )

        # Error estimate
        err_state = {}
        for key in state.keys():
            err_state[key] = new_state[key] - y4[key]

        err_norm = keras.ops.stack([keras.ops.norm(v, ord=2, axis=-1) for v in err_state.values()])
        err = keras.ops.max(err_norm)

        new_step_size = h * keras.ops.clip(0.9 * (tolerance / (err + 1e-12)) ** 0.2, 0.2, 5.0)
        new_step_size = keras.ops.clip(new_step_size, min_step_size, max_step_size)
    else:
        new_step_size = h

    new_time = time + h
    return new_state, new_time, new_step_size


def integrate_fixed(
    fn: Callable,
    state: dict[str, ArrayLike],
    start_time: ArrayLike,
    stop_time: ArrayLike,
    steps: int,
    method: str = "rk45",
    **kwargs,
) -> dict[str, ArrayLike]:
    if steps <= 0:
        raise ValueError("Number of steps must be positive.")

    match method:
        case "euler":
            step_fn = euler_step
        case "rk45":
            step_fn = rk45_step
        case "tsit5":
            step_fn = tsit5_step
        case str() as name:
            raise ValueError(f"Unknown integration method name: {name!r}")
        case other:
            raise TypeError(f"Invalid integration method: {other!r}")

    step_fn = partial(step_fn, fn, **kwargs, use_adaptive_step_size=False)
    step_size = (stop_time - start_time) / steps

    time = start_time

    def body(_loop_var, _loop_state):
        _state, _time = _loop_state
        _state, _time, _ = step_fn(_state, _time, step_size)

        return _state, _time

    state, time = keras.ops.fori_loop(0, steps, body, (state, time))

    return state


def integrate_adaptive(
    fn: Callable,
    state: dict[str, ArrayLike],
    start_time: ArrayLike,
    stop_time: ArrayLike,
    min_steps: int = 10,
    max_steps: int = 1000,
    method: str = "rk45",
    **kwargs,
) -> dict[str, ArrayLike]:
    if max_steps <= min_steps:
        raise ValueError("Maximum number of steps must be greater than minimum number of steps.")

    match method:
        case "euler":
            step_fn = euler_step
        case "rk45":
            step_fn = rk45_step
        case "tsit5":
            step_fn = tsit5_step
        case str() as name:
            raise ValueError(f"Unknown integration method name: {name!r}")
        case other:
            raise TypeError(f"Invalid integration method: {other!r}")

    step_fn = partial(step_fn, fn, **kwargs, use_adaptive_step_size=True)

    def cond(_state, _time, _step_size, _step):
        # while step < min_steps or time_remaining > 0 and step < max_steps

        # time remaining after the next step
        time_remaining = keras.ops.abs(stop_time - (_time + _step_size))

        return keras.ops.logical_or(
            keras.ops.all(_step < min_steps),
            keras.ops.logical_and(keras.ops.all(time_remaining > 0), keras.ops.all(_step < max_steps)),
        )

    def body(_state, _time, _step_size, _step):
        _step = _step + 1

        # time remaining after the next step
        time_remaining = stop_time - (_time + _step_size)

        min_step_size = time_remaining / (max_steps - _step)
        max_step_size = time_remaining / keras.ops.maximum(min_steps - _step, 1.0)

        # reorder
        min_step_size, max_step_size = (
            keras.ops.minimum(min_step_size, max_step_size),
            keras.ops.maximum(min_step_size, max_step_size),
        )

        _state, _time, _step_size = step_fn(
            _state, _time, _step_size, min_step_size=min_step_size, max_step_size=max_step_size
        )

        return _state, _time, _step_size, _step

    # select initial step size conservatively
    step_size = (stop_time - start_time) / max_steps

    step = 0
    time = start_time

    state, time, step_size, step = keras.ops.while_loop(cond, body, [state, time, step_size, step])

    # do the last step
    step_size = stop_time - time
    state, _, _ = step_fn(state, time, step_size)
    step = step + 1

    logging.debug("Finished integration after {} steps.", step)

    return state


def integrate_scheduled(
    fn: Callable,
    state: dict[str, ArrayLike],
    steps: Tensor | np.ndarray,
    method: str = "rk45",
    **kwargs,
) -> dict[str, ArrayLike]:
    match method:
        case "euler":
            step_fn = euler_step
        case "rk45":
            step_fn = rk45_step
        case "tsit5":
            step_fn = tsit5_step
        case str() as name:
            raise ValueError(f"Unknown integration method name: {name!r}")
        case other:
            raise TypeError(f"Invalid integration method: {other!r}")

    step_fn = partial(step_fn, fn, **kwargs, use_adaptive_step_size=False)

    def body(_loop_var, _loop_state):
        _time = steps[_loop_var]
        step_size = steps[_loop_var + 1] - steps[_loop_var]

        _loop_state, _, _ = step_fn(_loop_state, _time, step_size)
        return _loop_state

    state = keras.ops.fori_loop(0, len(steps) - 1, body, state)
    return state


def integrate_scipy(
    fn: Callable,
    state: dict[str, ArrayLike],
    start_time: ArrayLike,
    stop_time: ArrayLike,
    scipy_kwargs: dict | None = None,
    **kwargs,
) -> dict[str, ArrayLike]:
    import scipy.integrate

    scipy_kwargs = scipy_kwargs or {}
    keys = list(state.keys())
    # convert to tensor before determining the shape in case a number was passed
    shapes = keras.tree.map_structure(lambda x: keras.ops.shape(keras.ops.convert_to_tensor(x)), state)
    adapter = Adapter().concatenate(keys, into="x", axis=-1).convert_dtype(np.float32, np.float64)

    def state_to_vector(state):
        state = keras.tree.map_structure(keras.ops.convert_to_numpy, state)
        # flatten state
        state = keras.tree.map_structure(lambda x: keras.ops.reshape(x, (-1,)), state)
        # apply concatenation
        x = adapter.forward(state)["x"]
        return x

    def vector_to_state(x):
        state = adapter.inverse({"x": x})
        state = {key: keras.ops.reshape(value, shapes[key]) for key, value in state.items()}
        state = keras.tree.map_structure(keras.ops.convert_to_tensor, state)
        return state

    def scipy_wrapper_fn(time, x):
        state = vector_to_state(x)
        time = keras.ops.convert_to_tensor(time, dtype="float32")
        deltas = fn(time, **filter_kwargs(state, fn))
        return state_to_vector(deltas)

    result = scipy.integrate.solve_ivp(
        scipy_wrapper_fn,
        (start_time, stop_time),
        state_to_vector(state),
        **scipy_kwargs,
    )

    result = vector_to_state(result.y[:, -1])
    return result


def integrate(
    fn: Callable,
    state: dict[str, ArrayLike],
    start_time: ArrayLike | None = None,
    stop_time: ArrayLike | None = None,
    min_steps: int = 10,
    max_steps: int = 10_000,
    steps: int | Literal["adaptive"] | Tensor | np.ndarray = 100,
    method: str = "euler",
    **kwargs,
) -> dict[str, ArrayLike]:
    if isinstance(steps, str) and steps in ["adaptive", "dynamic"]:
        if start_time is None or stop_time is None:
            raise ValueError(
                "Please provide start_time and stop_time for the integration, was "
                f"'start_time={start_time}', 'stop_time={stop_time}'."
            )
        if method == "scipy":
            if min_steps != 10:
                warning("Setting min_steps has no effect for method 'scipy'")
            if max_steps != 10_000:
                warning("Setting max_steps has no effect for method 'scipy'")
            return integrate_scipy(fn, state, start_time, stop_time, **kwargs)
        return integrate_adaptive(fn, state, start_time, stop_time, min_steps, max_steps, method, **kwargs)
    elif isinstance(steps, int):
        if start_time is None or stop_time is None:
            raise ValueError(
                "Please provide start_time and stop_time for the integration, was "
                f"'start_time={start_time}', 'stop_time={stop_time}'."
            )
        return integrate_fixed(fn, state, start_time, stop_time, steps, method, **kwargs)
    elif isinstance(steps, Sequence) or isinstance(steps, np.ndarray) or keras.ops.is_tensor(steps):
        return integrate_scheduled(fn, state, steps, method, **kwargs)
    else:
        raise RuntimeError(f"Type or value of `steps` not understood (steps={steps})")


def euler_maruyama_step(
    drift_fn: Callable,
    diffusion_fn: Callable,
    state: dict[str, ArrayLike],
    time: ArrayLike,
    step_size: ArrayLike,
    noise: dict[str, ArrayLike],
) -> (dict[str, ArrayLike], ArrayLike, ArrayLike):
    """
    Performs a single Euler-Maruyama step for stochastic differential equations.

    Args:
        drift_fn: Function computing the drift term f(t, **state).
        diffusion_fn: Function computing the diffusion term g(t, **state).
        state: Current state, mapping variable names to tensors.
        time: Current time scalar tensor.
        step_size: Time increment dt.
        noise: Mapping of variable names to dW noise tensors.

    Returns:
        new_state: Updated state after one Euler-Maruyama step.
        new_time: time + dt.
    """
    # Compute drift and diffusion
    drift = drift_fn(time, **filter_kwargs(state, drift_fn))
    diffusion = diffusion_fn(time, **filter_kwargs(state, diffusion_fn))

    # Check noise keys
    if set(diffusion.keys()) != set(noise.keys()):
        raise ValueError("Keys of diffusion terms and noise do not match.")

    new_state = {}
    for key, d in drift.items():
        base = state[key] + step_size * d
        if key in diffusion:  # stochastic update
            base = base + diffusion[key] * noise[key]
        new_state[key] = base

    return new_state, time + step_size


def integrate_stochastic(
    drift_fn: Callable,
    diffusion_fn: Callable,
    state: dict[str, ArrayLike],
    start_time: ArrayLike,
    stop_time: ArrayLike,
    steps: int,
    seed: keras.random.SeedGenerator,
    method: str = "euler_maruyama",
    score_fn: Callable = None,
    corrector_steps: int = 0,
    noise_schedule=None,
    step_size_factor: float = 0.1,
    **kwargs,
) -> Union[dict[str, ArrayLike], tuple[dict[str, ArrayLike], dict[str, Sequence[ArrayLike]]]]:
    """
    Integrates a stochastic differential equation from start_time to stop_time.

    When score_fn is provided, performs predictor-corrector sampling where:
    - Predictor: reverse diffusion SDE solver
    - Corrector: annealed Langevin dynamics with step size e = sqrt(dim)

    Args:
        drift_fn: Function that computes the drift term.
        diffusion_fn: Function that computes the diffusion term.
        state: Dictionary containing the initial state.
        start_time: Starting time for integration.
        stop_time: Ending time for integration.
        steps: Number of integration steps.
        seed: Random seed for noise generation.
        method: Integration method to use, e.g., 'euler_maruyama'.
        score_fn: Optional score function for predictor-corrector sampling.
                 Should take (time, **state) and return score dict.
        corrector_steps: Number of corrector steps to take after each predictor step.
        noise_schedule: Noise schedule object for computing lambda_t and alpha_t in corrector.
        step_size_factor: Scaling factor for corrector step size.
        **kwargs: Additional arguments to pass to the step function.

    Returns:
        Final state dictionary after integration.
    """
    if steps <= 0:
        raise ValueError("Number of steps must be positive.")

    # Select step function based on method
    match method:
        case "euler_maruyama":
            step_fn = euler_maruyama_step
        case other:
            raise TypeError(f"Invalid integration method: {other!r}")

    # Prepare step function with partial application
    step_fn = partial(step_fn, drift_fn=drift_fn, diffusion_fn=diffusion_fn, **kwargs)

    # Time step
    step_size = (stop_time - start_time) / steps
    sqrt_dt = keras.ops.sqrt(keras.ops.abs(step_size))

    # Pre-generate noise history for predictor: shape = (steps, *state_shape)
    noise_history = {}
    for key, val in state.items():
        noise_history[key] = (
            keras.random.normal((steps, *keras.ops.shape(val)), dtype=keras.ops.dtype(val), seed=seed) * sqrt_dt
        )

    # Pre-generate corrector noise if score_fn is provided: shape = (steps, corrector_steps, *state_shape)
    corrector_noise_history = {}
    if corrector_steps > 0:
        if score_fn is None or noise_schedule is None:
            raise ValueError("Please provide both score_fn and noise_schedule when using corrector_steps > 0.")

        for key, val in state.items():
            corrector_noise_history[key] = keras.random.normal(
                (steps, corrector_steps, *keras.ops.shape(val)), dtype=keras.ops.dtype(val), seed=seed
            )

    def body(_loop_var, _loop_state):
        _current_state, _current_time = _loop_state
        _noise_i = {k: noise_history[k][_loop_var] for k in _current_state.keys()}

        # Predictor step
        new_state, new_time = step_fn(state=_current_state, time=_current_time, step_size=step_size, noise=_noise_i)

        # Corrector steps: annealed Langevin dynamics if score_fn is provided
        if corrector_steps > 0:
            for corrector_step in range(corrector_steps):
                score = score_fn(new_time, **filter_kwargs(new_state, score_fn))
                _corrector_noise = {k: corrector_noise_history[k][_loop_var, corrector_step] for k in new_state.keys()}

                # Compute noise schedule components for corrector step size
                log_snr_t = noise_schedule.get_log_snr(t=new_time, training=False)
                alpha_t, _ = noise_schedule.get_alpha_sigma(log_snr_t=log_snr_t)

                # Corrector update: x_i+1 = x_i + e * score + sqrt(2e) * noise_corrector
                # where e = 2*alpha_t * (r * ||z|| / ||score||)**2
                for k in new_state.keys():
                    if k in score:
                        z_norm = keras.ops.norm(_corrector_noise[k], axis=-1, keepdims=True)
                        score_norm = keras.ops.norm(score[k], axis=-1, keepdims=True)

                        # Prevent division by zero
                        score_norm = keras.ops.maximum(score_norm, 1e-8)

                        e = 2.0 * alpha_t * (step_size_factor * z_norm / score_norm) ** 2
                        sqrt_2e = keras.ops.sqrt(2.0 * e)

                        new_state[k] = new_state[k] + e * score[k] + sqrt_2e * _corrector_noise[k]

        return new_state, new_time

    final_state, final_time = keras.ops.fori_loop(0, steps, body, (state, start_time))
    return final_state
