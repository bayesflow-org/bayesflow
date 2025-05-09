from collections.abc import Callable, Sequence
from functools import partial

import keras

import numpy as np
from typing import Literal, Union

from bayesflow.types import Tensor
from bayesflow.utils import filter_kwargs

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
        intermediate_state = state.copy()
        for key, delta in k1.items():
            intermediate_state[key] = state[key] + step_size * delta

        k2 = fn(time + step_size, **filter_kwargs(intermediate_state, fn))

        # check all keys are equal
        if set(k1.keys()) != set(k2.keys()):
            raise ValueError("Keys of the deltas do not match. Please return zero for unchanged variables.")

        # compute next step size
        intermediate_error = keras.ops.stack([keras.ops.norm(k2[key] - k1[key], ord=2, axis=-1) for key in k1])
        new_step_size = step_size * tolerance / (intermediate_error + 1e-9)

        new_step_size = keras.ops.clip(new_step_size, min_step_size, max_step_size)

        # consolidate step size
        new_step_size = keras.ops.take(new_step_size, keras.ops.argmin(keras.ops.abs(new_step_size)))
    else:
        new_step_size = step_size

    # apply updates
    new_state = state.copy()
    for key in k1.keys():
        new_state[key] = state[key] + step_size * k1[key]

    new_time = time + step_size

    return new_state, new_time, new_step_size


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
    step_size = last_step_size

    k1 = fn(time, **filter_kwargs(state, fn))

    intermediate_state = state.copy()
    for key, delta in k1.items():
        intermediate_state[key] = state[key] + 0.5 * step_size * delta

    k2 = fn(time + 0.5 * step_size, **filter_kwargs(intermediate_state, fn))

    intermediate_state = state.copy()
    for key, delta in k2.items():
        intermediate_state[key] = state[key] + 0.5 * step_size * delta

    k3 = fn(time + 0.5 * step_size, **filter_kwargs(intermediate_state, fn))

    intermediate_state = state.copy()
    for key, delta in k3.items():
        intermediate_state[key] = state[key] + step_size * delta

    k4 = fn(time + step_size, **filter_kwargs(intermediate_state, fn))

    if use_adaptive_step_size:
        intermediate_state = state.copy()
        for key, delta in k4.items():
            intermediate_state[key] = state[key] + 0.5 * step_size * delta

        k5 = fn(time + 0.5 * step_size, **filter_kwargs(intermediate_state, fn))

        # check all keys are equal
        if not all(set(k.keys()) == set(k1.keys()) for k in [k2, k3, k4, k5]):
            raise ValueError("Keys of the deltas do not match. Please return zero for unchanged variables.")

        # compute next step size
        intermediate_error = keras.ops.stack([keras.ops.norm(k5[key] - k4[key], ord=2, axis=-1) for key in k5.keys()])
        new_step_size = step_size * tolerance / (intermediate_error + 1e-9)

        new_step_size = keras.ops.clip(new_step_size, min_step_size, max_step_size)

        # consolidate step size
        new_step_size = keras.ops.take(new_step_size, keras.ops.argmin(keras.ops.abs(new_step_size)))
    else:
        new_step_size = step_size

    # apply updates
    new_state = state.copy()
    for key in k1.keys():
        new_state[key] = state[key] + (step_size / 6.0) * (k1[key] + 2.0 * k2[key] + 2.0 * k3[key] + k4[key])

    new_time = time + step_size

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
    **kwargs,
) -> Union[dict[str, ArrayLike], tuple[dict[str, ArrayLike], dict[str, Sequence[ArrayLike]]]]:
    """
    Integrates a stochastic differential equation from start_time to stop_time.

    Args:
        drift_fn: Function that computes the drift term.
        diffusion_fn: Function that computes the diffusion term.
        state: Dictionary containing the initial state.
        start_time: Starting time for integration.
        stop_time: Ending time for integration.
        steps: Number of integration steps.
        seed: Random seed for noise generation.
        method: Integration method to use, e.g., 'euler_maruyama'.
        **kwargs: Additional arguments to pass to the step function.

    Returns:
        If return_noise is False, returns the final state dictionary.
        If return_noise is True, returns a tuple of (final_state, noise_history).
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

    # Pre-generate noise history: shape = (steps, *state_shape)
    noise_history = {}
    for key, val in state.items():
        noise_history[key] = (
            keras.random.normal((steps, *keras.ops.shape(val)), dtype=keras.ops.dtype(val), seed=seed) * sqrt_dt
        )

    def body(_loop_var, _loop_state):
        _current_state, _current_time = _loop_state
        _noise_i = {k: noise_history[k][_loop_var] for k in _current_state.keys()}
        new_state, new_time = step_fn(state=_current_state, time=_current_time, step_size=step_size, noise=_noise_i)
        return new_state, new_time

    final_state, final_time = keras.ops.fori_loop(0, steps, body, (state, start_time))
    return final_state
