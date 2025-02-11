from collections.abc import Callable
from functools import partial

import keras

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


def integrate(
    fn: Callable,
    state: dict[str, ArrayLike],
    start_time: ArrayLike,
    stop_time: ArrayLike,
    min_steps: int = 10,
    max_steps: int = 10_000,
    steps: int = "adaptive",
    method: str = "rk45",
    **kwargs,
) -> dict[str, ArrayLike]:
    match steps:
        case "adaptive" | "dynamic":
            return integrate_adaptive(fn, state, start_time, stop_time, min_steps, max_steps, method, **kwargs)
        case int():
            return integrate_fixed(fn, state, start_time, stop_time, steps, method, **kwargs)
        case _:
            raise RuntimeError("Type or value of `steps` not understood.")
