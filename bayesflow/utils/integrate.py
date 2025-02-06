from collections.abc import Callable
from functools import partial
from numbers import Number

import keras

from bayesflow.types import Tensor
from . import logging

ArrayLike = int | float | Tensor


from bayesflow.utils import filter_kwargs


def fixed_euler(
    fn: Callable, state: dict[str, ArrayLike], time: ArrayLike, step_size: ArrayLike
) -> (dict[str, ArrayLike], ArrayLike, ArrayLike):
    state_deltas = fn(time, **filter_kwargs(state, fn))

    for key, value in state_deltas.items():
        state[key] += step_size * value

    time += step_size

    return state, time, step_size


def dynamic_euler(
    fn: Callable, state: dict[str, ArrayLike], time: ArrayLike, step_size: ArrayLike, tolerance: ArrayLike = 1e-3
) -> (dict[str, ArrayLike], ArrayLike, ArrayLike):
    deltas = fn(time, **filter_kwargs(state, fn))

    intermediate_state = state.copy()
    for key, delta in deltas.items():
        intermediate_state[key] += step_size * delta

    intermediate_deltas = fn(time + step_size, **filter_kwargs(intermediate_state, fn))

    # check all keys are equal
    if set(deltas.keys()) != set(intermediate_deltas.keys()):
        raise ValueError("Keys of the deltas do not match. Please return zero for unchanged variables.")

    # apply updates
    for key, value in state.items():
        state[key] += step_size * value

    time += step_size

    # compute next step size
    intermediate_error = keras.ops.stack(
        [keras.ops.norm(intermediate_deltas[key] - deltas[key], ord=2, axis=-1) for key in deltas]
    )
    step_size = step_size * tolerance / intermediate_error

    # consolidate step size
    step_size = keras.ops.min(step_size)

    return state, time, step_size


def fixed_rk45(
    fn: Callable, state: dict[str, ArrayLike], time: ArrayLike, step_size: ArrayLike
) -> (dict[str, ArrayLike], ArrayLike, ArrayLike):
    k1 = fn(time, **filter_kwargs(state, fn))

    # TODO: this can probably be done nicer somehow (notably not with a loop)
    intermediate_state = state.copy()
    for key, delta in k1.items():
        intermediate_state[key] += 0.5 * step_size * delta

    k2 = fn(time + 0.5 * step_size, **filter_kwargs(intermediate_state, fn))

    intermediate_state = state.copy()
    for key, delta in k2.items():
        intermediate_state[key] += 0.5 * step_size * delta

    k3 = fn(time + 0.5 * step_size, **filter_kwargs(intermediate_state, fn))

    intermediate_state = state.copy()
    for key, delta in k3.items():
        intermediate_state[key] += step_size * delta

    k4 = fn(time + step_size, **filter_kwargs(intermediate_state, fn))

    # check all keys are equal
    if not all(set(k.keys()) == set(k1.keys()) for k in [k2, k3, k4]):
        raise ValueError("Keys of the deltas do not match. Please return zero for unchanged variables.")

    # apply updates
    for key in k4.keys():
        state[key] += (step_size / 6.0) * (k1[key] + 2.0 * k2[key] + 2.0 * k3[key] + k4[key])

    time += step_size

    return state, time, step_size


def dynamic_rk45(
    fn: Callable, state: dict[str, ArrayLike], time: ArrayLike, last_step_size: ArrayLike, tolerance: ArrayLike = 1e-3
) -> (dict[str, ArrayLike], ArrayLike, ArrayLike):
    step_size = last_step_size

    k1 = fn(time, **filter_kwargs(state, fn))

    intermediate_state = state.copy()
    for key, delta in k1.items():
        intermediate_state[key] += 0.5 * step_size * delta

    k2 = fn(time + 0.5 * step_size, **filter_kwargs(intermediate_state, fn))

    intermediate_state = state.copy()
    for key, delta in k2.items():
        intermediate_state[key] += 0.5 * step_size * delta

    k3 = fn(time + 0.5 * step_size, **filter_kwargs(intermediate_state, fn))

    intermediate_state = state.copy()
    for key, delta in k3.items():
        intermediate_state[key] += step_size * delta

    k4 = fn(time + step_size, **filter_kwargs(intermediate_state, fn))

    intermediate_state = state.copy()
    for key, delta in k4.items():
        intermediate_state[key] += 0.5 * step_size * delta

    k5 = fn(time + 0.5 * step_size, **filter_kwargs(intermediate_state, fn))

    # check all keys are equal
    if not all(set(k.keys()) == set(k1.keys()) for k in [k2, k3, k4, k5]):
        raise ValueError("Keys of the deltas do not match. Please return zero for unchanged variables.")

    # apply updates
    for key in k5.keys():
        state[key] += (step_size / 6.0) * (k1[key] + 2.0 * k2[key] + 2.0 * k3[key] + k4[key])

    time += step_size

    # compute next step size
    intermediate_error = keras.ops.stack([keras.ops.norm(k5[key] - k4[key], ord=2, axis=-1) for key in k5.keys()])
    step_size = step_size * tolerance / intermediate_error

    # consolidate step size
    step_size = keras.ops.min(step_size)

    return state, time, step_size


def step_function_factory(fn: Callable, method: str, start_time, stop_time, steps, step_size):
    match steps, step_size:
        case "adaptive", "adaptive":
            # TODO: select a better initial step size
            step_size = 1e-3
            use_adaptive_step_size = True
        case int(), "adaptive":
            step_size = (stop_time - start_time) / steps
            use_adaptive_step_size = False
        case "adaptive", Number():
            use_adaptive_step_size = False
        case int(), Number():
            raise ValueError("Cannot specify both `steps` and `step_size`.")
        case _:
            raise RuntimeError("Type or value of `steps` or `step_size` not understood.")

    match method:
        case "euler":
            if use_adaptive_step_size:
                step_fn = dynamic_euler
            else:
                step_fn = fixed_euler
        case "rk45":
            if use_adaptive_step_size:
                step_fn = dynamic_rk45
            else:
                step_fn = fixed_rk45
        case str() as name:
            raise ValueError(f"Unknown integration method name: {name!r}")
        case other:
            raise TypeError(f"Invalid integration method: {other!r}")

    step_fn = partial(step_fn, fn)

    return step_fn, step_size


def integrate(
    fn: Callable,
    state: dict[str, ArrayLike],
    start_time: ArrayLike,
    stop_time: ArrayLike,
    steps: int = "adaptive",
    step_size: Number = "adaptive",
    method: str = "rk45",
    **kwargs,
) -> dict[str, ArrayLike]:
    step_fn, step_size = step_function_factory(fn, method, start_time, stop_time, steps, step_size)

    if kwargs:
        step_fn = partial(step_fn, **kwargs)

    def cond(state, time, step_size, step):
        # step until the next step would exceed the stop time
        return keras.ops.all(time + step_size < stop_time)

    def body(state, time, step_size, step):
        state, time, step_size = step_fn(state, time, step_size)
        return state, time, step_size, step + 1

    step = 0
    time = start_time

    state, time, step_size, step = keras.ops.while_loop(cond, body, [state, time, step_size, step])

    # do the last step
    step_size = stop_time - time
    state, _, _ = step_fn(state, time, step_size)
    step += 1

    logging.debug("Finished integration after {} steps.", step)

    return state
