from collections.abc import Callable, Sequence
from typing import Dict, Tuple, Optional
from functools import partial

import keras

import numpy as np
from typing import Literal, Union

from bayesflow.adapters import Adapter
from bayesflow.types import Tensor
from bayesflow.utils import filter_kwargs
from bayesflow.utils.logging import warning, debug


ArrayLike = int | float | Tensor
StateDict = Dict[str, ArrayLike]


DETERMINISTIC_METHODS = ["euler", "rk45", "tsit5"]
STOCHASTIC_METHODS = ["euler_maruyama", "sea", "shark", "two_step_adaptive", "langevin", "glass"]


def _check_all_nans(state: StateDict):
    all_nans_flags = []
    for v in state.values():
        all_nans_flags.append(keras.ops.all(keras.ops.isnan(v)))
    return keras.ops.all(keras.ops.stack(all_nans_flags))


def _as_concrete_bool(x) -> bool | None:
    """Return ``bool(x)`` if `x` holds a concrete value, or None if `x` is symbolic,
    e.g. while tracing inside ``tf.function`` or ``jax.jit``."""
    try:
        return bool(x)
    except Exception:
        return None


def euler_step(
    fn: Callable,
    state: StateDict,
    time: ArrayLike,
    step_size: ArrayLike,
) -> Tuple[StateDict, ArrayLike, None, ArrayLike]:
    k1 = fn(time, **filter_kwargs(state, fn))

    new_state = state.copy()
    for key in k1.keys():
        new_state[key] = state[key] + step_size * k1[key]
    new_time = time + step_size

    return new_state, new_time, None, 0.0


def add_scaled(state, ks, coeffs, h):
    out = {}
    for key, y in state.items():
        acc = keras.ops.zeros_like(y)
        for c, k in zip(coeffs, ks):
            acc = acc + c * k[key]
        out[key] = y + h * acc
    return out


def _error_ratio(
    err_state: StateDict,
    state: StateDict,
    new_state: StateDict,
    atol: ArrayLike,
    rtol: ArrayLike,
) -> ArrayLike:
    """Normalize a local error estimate by a per-component tolerance.

    The error of an entry is reduced with a root-mean-square norm, which keeps the criterion independent of
    the number of components.

    Parameters
    ----------
    err_state : dict
        Local error estimate per state entry, as returned by the adaptive step functions.
    state : dict
        State at the beginning of the step.
    new_state : dict
        State at the end of the step.
    atol : int or float or Tensor
        Absolute tolerance.
    rtol : int or float or Tensor
        Relative tolerance.

    Returns
    -------
    int or float or Tensor
        The normalized error, which is at most 1 if the step meets the requested tolerance.
    """
    ratio = None
    for key, err in err_state.items():
        scale = atol + rtol * keras.ops.maximum(keras.ops.abs(state[key]), keras.ops.abs(new_state[key]))
        key_ratio = keras.ops.max(keras.ops.sqrt(keras.ops.mean(keras.ops.square(err / scale), axis=-1)))
        ratio = key_ratio if ratio is None else keras.ops.maximum(ratio, key_ratio)

    return ratio


def rk45_step(
    fn: Callable,
    state: StateDict,
    time: ArrayLike,
    step_size: ArrayLike,
    k1: StateDict = None,
    use_adaptive_step_size: bool = True,
) -> Tuple[StateDict, ArrayLike, StateDict | None, StateDict | None]:
    """
    Dormand-Prince 5(4) method with embedded error estimation [1].

    Dormand (1996), Numerical Methods for Differential Equations: A Computational Approach
    """
    h = step_size

    if k1 is None:  # reuse k1 if available
        k1 = fn(time, **filter_kwargs(state, fn))
    k2 = fn(time + h * (1 / 5), **filter_kwargs(add_scaled(state, [k1], [1 / 5], h), fn))
    k3 = fn(time + h * (3 / 10), **filter_kwargs(add_scaled(state, [k1, k2], [3 / 40, 9 / 40], h), fn))
    k4 = fn(time + h * (4 / 5), **filter_kwargs(add_scaled(state, [k1, k2, k3], [44 / 45, -56 / 15, 32 / 9], h), fn))
    k5 = fn(
        time + h * (8 / 9),
        **filter_kwargs(
            add_scaled(state, [k1, k2, k3, k4], [19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729], h), fn
        ),
    )
    k6 = fn(
        time + h,
        **filter_kwargs(
            add_scaled(state, [k1, k2, k3, k4, k5], [9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656], h),
            fn,
        ),
    )

    # 5th order solution
    new_state = {}
    for key in k1.keys():
        new_state[key] = state[key] + h * (
            35 / 384 * k1[key] + 500 / 1113 * k3[key] + 125 / 192 * k4[key] - 2187 / 6784 * k5[key] + 11 / 84 * k6[key]
        )

    new_time = time + h
    if not use_adaptive_step_size:
        return new_state, new_time, None, None

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

    return new_state, new_time, k7, err_state


def tsit5_step(
    fn: Callable,
    state: StateDict,
    time: ArrayLike,
    step_size: ArrayLike,
    k1: StateDict = None,
    use_adaptive_step_size: bool = True,
) -> Tuple[StateDict, ArrayLike, StateDict | None, StateDict | None]:
    """
    Implements a single step of the Tsitouras 5/4 Runge-Kutta method [1].

    [1] Tsitouras (2011), Runge--Kutta pairs of order 5(4) satisfying only the first column simplifying assumption
    """
    h = step_size

    # Butcher tableau coefficients
    c2 = 0.161
    c3 = 0.327
    c4 = 0.9
    c5 = 0.9800255409045097

    if k1 is None:  # reuse k1 if available
        k1 = fn(time, **filter_kwargs(state, fn))
    k2 = fn(time + h * c2, **filter_kwargs(add_scaled(state, [k1], [0.161], h), fn))
    k3 = fn(
        time + h * c3, **filter_kwargs(add_scaled(state, [k1, k2], [-0.0084806554923570, 0.3354806554923570], h), fn)
    )
    k4 = fn(
        time + h * c4,
        **filter_kwargs(
            add_scaled(state, [k1, k2, k3], [2.897153057105494, -6.359448489975075, 4.362295432869581], h), fn
        ),
    )
    k5 = fn(
        time + h * c5,
        **filter_kwargs(
            add_scaled(
                state,
                [k1, k2, k3, k4],
                [5.325864828439257, -11.74888356406283, 7.495539342889836, -0.09249506636175525],
                h,
            ),
            fn,
        ),
    )
    k6 = fn(
        time + h,
        **filter_kwargs(
            add_scaled(
                state,
                [k1, k2, k3, k4, k5],
                [5.86145544294270, -12.92096931784711, 8.159367898576159, -0.07158497328140100, -0.02826905039406838],
                h,
            ),
            fn,
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

    new_time = time + h
    if not use_adaptive_step_size:
        return new_state, new_time, None, None

    k7 = fn(time + h, **filter_kwargs(new_state, fn))

    err_state = {}
    for key in state.keys():
        err_state[key] = h * (
            0.007880878010261995 * k3[key]
            + 0.5823571654525552 * k5[key]
            + 0.015151515151515152 * k7[key]
            - 0.00178001105222577714 * k1[key]
            - 0.0008164344596567469 * k2[key]
            - 0.1447110071732629 * k4[key]
            - 0.45808210592918697 * k6[key]
        )

    return new_state, new_time, k7, err_state


def integrate_fixed(
    fn: Callable,
    state: StateDict,
    start_time: ArrayLike,
    stop_time: ArrayLike,
    steps: int,
    method: str,
    **kwargs,
) -> StateDict:
    if steps <= 0:
        raise ValueError("Number of steps must be positive.")

    match method:
        case "euler":
            step_fn = partial(euler_step, fn, **filter_kwargs(kwargs, euler_step))
        case "rk45":
            step_fn = partial(rk45_step, fn, **filter_kwargs(kwargs, rk45_step), use_adaptive_step_size=False)
        case "tsit5":
            step_fn = partial(tsit5_step, fn, **filter_kwargs(kwargs, tsit5_step), use_adaptive_step_size=False)
        case str() as name:
            raise ValueError(f"Unknown integration method name: {name!r}")
        case other:
            raise TypeError(f"Invalid integration method: {other!r}")

    step_size = (stop_time - start_time) / steps

    def body(_loop_var, _loop_state):
        _state, _time = _loop_state
        _state, _time, _, _ = step_fn(_state, _time, step_size)
        return _state, _time

    state, _ = keras.ops.fori_loop(
        0,
        steps,
        body,
        (state, start_time),
    )
    return state


def integrate_scheduled(
    fn: Callable,
    state: StateDict,
    steps: Tensor | np.ndarray,
    method: str,
    **kwargs,
) -> StateDict:
    steps = keras.ops.convert_to_tensor(steps, dtype=keras.config.floatx())

    match method:
        case "euler":
            step_fn = partial(euler_step, fn, **filter_kwargs(kwargs, euler_step))
        case "rk45":
            step_fn = partial(rk45_step, fn, **filter_kwargs(kwargs, rk45_step), use_adaptive_step_size=False)
        case "tsit5":
            step_fn = partial(tsit5_step, fn, **filter_kwargs(kwargs, tsit5_step), use_adaptive_step_size=False)
        case str() as name:
            raise ValueError(f"Unknown integration method name: {name!r}")
        case other:
            raise TypeError(f"Invalid integration method: {other!r}")

    def body(_loop_var, _loop_state):
        _time = steps[_loop_var]
        step_size = steps[_loop_var + 1] - steps[_loop_var]
        _loop_state, _, _, _ = step_fn(_loop_state, _time, step_size)
        return _loop_state

    state = keras.ops.fori_loop(
        0,
        keras.ops.shape(steps)[0] - 1,
        body,
        state,
    )
    return state


def integrate_adaptive(
    fn: Callable,
    state: dict[str, ArrayLike],
    start_time: ArrayLike,
    stop_time: ArrayLike,
    min_steps: int,
    max_steps: int,
    method: str,
    **kwargs,
) -> dict[str, ArrayLike]:
    if max_steps <= min_steps:
        raise ValueError("Maximum number of steps must be greater than minimum number of steps.")

    match method:
        case "rk45":
            step_fn = partial(rk45_step, fn, **filter_kwargs(kwargs, rk45_step), use_adaptive_step_size=True)
        case "tsit5":
            step_fn = partial(tsit5_step, fn, **filter_kwargs(kwargs, tsit5_step), use_adaptive_step_size=True)
        case "euler":
            raise ValueError("Adaptive step sizing is not supported for the 'euler' method.")
        case str() as name:
            raise ValueError(f"Unknown integration method name: {name!r}")
        case other:
            raise TypeError(f"Invalid integration method: {other!r}")

    # density computation (state carries more than one entry) needs higher accuracy than sampling,
    atol = keras.ops.convert_to_tensor(kwargs.get("atol", 1e-6), dtype="float32")
    rtol = keras.ops.convert_to_tensor(kwargs.get("rtol", 1e-4 if len(state) == 1 else 1e-5), dtype="float32")
    initial_step = keras.ops.convert_to_tensor((stop_time - start_time) / float(min_steps), dtype="float32")
    step0 = keras.ops.convert_to_tensor(0.0, dtype="float32")
    count_not_accepted = keras.ops.convert_to_tensor(0.0, dtype="float32")

    # "First Same As Last" (FSAL) property
    k1_0 = fn(start_time, **filter_kwargs(state, fn))

    def cond(_state, _time, _step_size, _step, _k1, _count_not_accepted):
        time_remaining = keras.ops.sign(stop_time - start_time) * (stop_time - (_time + _step_size))
        step_lt_min = keras.ops.less(_step, float(min_steps))
        step_lt_max = keras.ops.less(_step, float(max_steps))

        all_nans = _check_all_nans(_state)

        end_now = keras.ops.logical_or(
            step_lt_min, keras.ops.logical_and(keras.ops.all(time_remaining > 0), step_lt_max)
        )
        return keras.ops.logical_and(~all_nans, end_now)

    def body(_state, _time, _step_size, _step, _k1, _count_not_accepted):
        # Time remaining from current point
        time_remaining = keras.ops.abs(stop_time - _time)
        min_step_size = time_remaining / (max_steps - _step)
        max_step_size = time_remaining / keras.ops.maximum(min_steps - _step, 1.0)
        h = keras.ops.sign(_step_size) * keras.ops.clip(keras.ops.abs(_step_size), min_step_size, max_step_size)

        # Take one trial step
        new_state, new_time, new_k1, err_state = step_fn(
            state=_state,
            time=_time,
            step_size=h,
            k1=_k1,
        )

        # New step size suggestion
        error_ratio = _error_ratio(err_state, _state, new_state, atol, rtol)
        new_step_size = h * keras.ops.clip(0.9 * (1.0 / (error_ratio + 1e-12)) ** 0.2, 0.2, 5.0)
        new_step_size = keras.ops.sign(new_step_size) * keras.ops.clip(
            keras.ops.abs(new_step_size), min_step_size, max_step_size
        )

        # Error control
        within_tol = keras.ops.logical_or(
            keras.ops.less_equal(error_ratio, 1.0),
            keras.ops.isnan(error_ratio),
        )
        at_min = keras.ops.less_equal(
            keras.ops.abs(h),
            keras.ops.abs(min_step_size),
        )
        accepted = keras.ops.logical_or(within_tol, at_min)

        updated_state = keras.ops.cond(accepted, lambda: new_state, lambda: _state)
        updated_time = keras.ops.cond(accepted, lambda: new_time, lambda: _time)
        updated_k1 = keras.ops.cond(accepted, lambda: new_k1, lambda: _k1)

        # Step counter: increment only on accepted steps
        updated_step = _step + keras.ops.where(accepted, 1.0, 0.0)
        _count_not_accepted = _count_not_accepted + keras.ops.where(accepted, 0.0, 1.0)

        # For the next iteration, always use the new suggested step size
        return updated_state, updated_time, new_step_size, updated_step, updated_k1, _count_not_accepted

    # Run the adaptive loop
    state, time, step_size, step, k1, count_not_accepted = keras.ops.while_loop(
        cond,
        body,
        [state, start_time, initial_step, step0, k1_0, count_not_accepted],
        # never reached in normal operation and only guards against non-terminating loops (e.g. NaN pathologies)
        maximum_iterations=10 * max_steps,
    )

    # skipped while tracing (symbolic tensors have no concrete bool)
    all_nans = _as_concrete_bool(_check_all_nans(state))
    if all_nans:
        raise RuntimeError(f"All values are NaNs in state during integration at {time}.")

    # Final step to hit stop_time exactly
    time_diff = stop_time - time
    time_remaining = keras.ops.sign(stop_time - start_time) * time_diff

    def take_final_step():
        final_state, _, _, _ = step_fn(
            state=state,
            time=time,
            step_size=time_diff,
            k1=k1,
        )
        return final_state, step + 1.0

    state, step = keras.ops.cond(keras.ops.all(time_remaining > 0), take_final_step, lambda: (state, step))

    if all_nans is not None:  # not tracing, values are concrete
        debug(f"Finished integration after {step} steps with {count_not_accepted} rejected steps.")
    return state


def integrate_scipy(
    fn: Callable,
    state: StateDict,
    start_time: ArrayLike,
    stop_time: ArrayLike,
    scipy_kwargs: dict | None = None,
) -> StateDict:
    import scipy.integrate

    scipy_kwargs = scipy_kwargs or {}
    keys = list(state.keys())
    # convert to tensor before determining the shape in case a number was passed
    shapes = keras.tree.map_structure(lambda x: keras.ops.shape(keras.ops.convert_to_tensor(x)), state)
    adapter = Adapter().concatenate(keys, into="x", axis=-1).convert_dtype(np.float32, np.float64)

    def state_to_vector(state):
        state = keras.tree.map_structure(keras.ops.convert_to_numpy, state)
        # flatten state, staying in numpy
        state = keras.tree.map_structure(lambda x: np.reshape(x, (-1,)), state)
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


def _compile_loop_integrator(func: Callable) -> Callable:
    """On the TensorFlow backend, wrap an integration call in ``tf.function``.

    TensorFlow executes ``keras.ops.while_loop``/``fori_loop`` eagerly, one Python
    iteration per solver step. Tracing the whole integration once removes the per-step
    Python and GradientTape overhead.
    """
    if keras.backend.backend() != "tensorflow":
        return func

    import tensorflow as tf

    if not tf.executing_eagerly():
        return func

    return tf.function(func, autograph=False)


def integrate(
    fn: Callable,
    state: StateDict,
    start_time: ArrayLike | None = None,
    stop_time: ArrayLike | None = None,
    min_steps: int = 50,
    max_steps: int = 10_000,
    steps: int | Literal["adaptive"] | Tensor | np.ndarray = 100,
    method: str = "rk45",
    **kwargs,
) -> StateDict:
    if isinstance(steps, str) and steps not in ["adaptive", "dynamic"]:
        raise ValueError(f"Unknown steps value: {steps!r}. Use an integer, an array of times, or 'adaptive'.")
    if isinstance(steps, str) and steps in ["adaptive", "dynamic"]:
        if start_time is None or stop_time is None:
            raise ValueError(
                "Please provide start_time and stop_time for the integration, was "
                f"'start_time={start_time}', 'stop_time={stop_time}'."
            )
        if method == "scipy":
            if min_steps != 50:
                warning("Setting min_steps has no effect for method 'scipy'")
            if max_steps != 10_000:
                warning("Setting max_steps has no effect for method 'scipy'")
            return integrate_scipy(fn, state, start_time, stop_time, scipy_kwargs=kwargs.get("scipy_kwargs"))

        def run():
            return integrate_adaptive(fn, state, start_time, stop_time, min_steps, max_steps, method, **kwargs)

        result = _compile_loop_integrator(run)()
        # covers the TensorFlow backend, where the check inside integrate_adaptive is skipped while tracing
        if _as_concrete_bool(_check_all_nans(result)):
            raise RuntimeError("All values are NaNs in state during integration.")
        return result
    elif isinstance(steps, int):
        if start_time is None or stop_time is None:
            raise ValueError(
                "Please provide start_time and stop_time for the integration, was "
                f"'start_time={start_time}', 'stop_time={stop_time}'."
            )

        def run():
            return integrate_fixed(fn, state, start_time, stop_time, steps, method, **kwargs)

        return _compile_loop_integrator(run)()
    elif isinstance(steps, Sequence) or isinstance(steps, np.ndarray) or keras.ops.is_tensor(steps):

        def run():
            return integrate_scheduled(fn, state, steps, method, **kwargs)

        return _compile_loop_integrator(run)()
    else:
        raise RuntimeError(f"Type or value of `steps` not understood (steps={steps})")


############ SDE Solvers #############


def generate_noise(z: StateDict, seed: keras.random.SeedGenerator) -> StateDict:
    noise = {
        k: keras.random.normal(keras.ops.shape(val), dtype=keras.ops.dtype(val), seed=seed) for k, val in z.items()
    }
    return noise


def stochastic_adaptive_step_size_controller(
    state,
    drift,
    adaptive_factor: ArrayLike,
    min_step_size: float = -float("inf"),
    max_step_size: float = float("inf"),
) -> ArrayLike:
    """
    Adaptive step size controller based on [1]. Similar to a tamed explicit Euler method when used in Euler-Maruyama.

    Adaptive step sizing uses:
        h = max(1, ||x||**2) / max(1, ||f(x)||**2) * adaptive_factor


    [1] Fang & Giles, Adaptive Euler-Maruyama Method for SDEs with Non-Globally Lipschitz Drift Coefficients (2020)

    Returns
    -------
    New step size.
    """
    state_norms = []
    drift_norms = []
    for key in state.keys():
        state_norms.append(keras.ops.norm(state[key], ord=2, axis=-1))
        drift_norms.append(keras.ops.norm(drift[key], ord=2, axis=-1))
    state_norm = keras.ops.stack(state_norms)
    drift_norm = keras.ops.stack(drift_norms)
    max_state_norm = keras.ops.maximum(
        keras.ops.cast(1.0, dtype=keras.ops.dtype(state_norm)), keras.ops.max(state_norm) ** 2
    )
    max_drift_norm = keras.ops.maximum(
        keras.ops.cast(1.0, dtype=keras.ops.dtype(drift_norm)), keras.ops.max(drift_norm) ** 2
    )
    new_step_size = max_state_norm / max_drift_norm * adaptive_factor
    new_step_size = keras.ops.clip(new_step_size, min_step_size, max_step_size)
    return new_step_size


def euler_maruyama_step(
    drift_fn: Callable,
    diffusion_fn: Callable,
    state: StateDict,
    time: ArrayLike,
    step_size: ArrayLike,
    noise: StateDict,
    use_adaptive_step_size: bool = False,
    min_step_size: float = -float("inf"),
    max_step_size: float = float("inf"),
) -> Union[Tuple[StateDict, ArrayLike, ArrayLike], Tuple[StateDict, ArrayLike, ArrayLike, StateDict]]:
    """
    Performs a single Euler-Maruyama step for stochastic differential equations.

    Args:
        drift_fn: Function computing the drift term f(t, **state).
        diffusion_fn: Function computing the diffusion term g(t, **state).
        state: Current state, mapping variable names to tensors.
        time: Current time scalar tensor.
        step_size: Time increment dt.
        noise: Mapping of variable names to dW noise tensors.
        use_adaptive_step_size: Whether to use adaptive step sizing.
        min_step_size: Minimum allowed step size.
        max_step_size: Maximum allowed step size.

    Returns:
        new_state: Updated state after one Euler-Maruyama step.
        new_time: time + dt.
    """
    # Compute drift and diffusion
    drift = drift_fn(time, **filter_kwargs(state, drift_fn))
    diffusion = diffusion_fn(time, **filter_kwargs(state, diffusion_fn))

    new_step_size = step_size
    if use_adaptive_step_size:
        sign_step = keras.ops.sign(step_size)
        new_step_size = stochastic_adaptive_step_size_controller(
            state=state,
            drift=drift,
            adaptive_factor=max_step_size,
            min_step_size=min_step_size,
            max_step_size=max_step_size,
        )
        new_step_size = sign_step * keras.ops.abs(new_step_size)

    sqrt_step_size = keras.ops.sqrt(keras.ops.abs(new_step_size))

    new_state = {}
    for key, d in drift.items():
        base = state[key] + new_step_size * d
        if key in diffusion:
            base = base + diffusion[key] * sqrt_step_size * noise[key]
        new_state[key] = base

    if use_adaptive_step_size:
        return new_state, time + new_step_size, new_step_size, state
    return new_state, time + new_step_size, new_step_size


def two_step_adaptive_step(
    drift_fn: Callable,
    diffusion_fn: Callable,
    state: StateDict,
    time: ArrayLike,
    step_size: ArrayLike,
    noise: StateDict,
    last_state: StateDict = None,
    use_adaptive_step_size: bool = True,
    min_step_size: float = -float("inf"),
    max_step_size: float = float("inf"),
    e_rel: float = 0.1,
    e_abs: float = None,
    r: float = 0.9,
    adapt_safety: float = 0.9,
) -> Union[
    Tuple[StateDict, ArrayLike, ArrayLike],
    Tuple[StateDict, ArrayLike, ArrayLike, StateDict],
]:
    """
    Performs a single adaptive step for stochastic differential equations based on [1].

    Based on

    This method uses a predictor-corrector approach with error estimation:
    1. Take an Euler-Maruyama step (predictor)
    2. Take another Euler-Maruyama step from the predicted state
    3. Average the two predictions (corrector)
    4. Estimate error and adapt step size

    When step_size reaches min_step_size, steps are always accepted regardless of
    error to ensure progress and termination within max_steps.

    [1] Jolicoeur-Martineau et al. (2021) "Gotta Go Fast When Generating Data with Score-Based Models"

    Args:
        drift_fn: Function computing the drift term f(t, **state).
        diffusion_fn: Function computing the diffusion term g(t, **state).
        state: Current state, mapping variable names to tensors.
        time: Current time scalar tensor.
        step_size: Time increment dt.
        noise: Mapping of variable names to dW noise tensors (pre-scaled by sqrt(dt)).
        last_state: Previous state for error estimation.
        use_adaptive_step_size: Whether to adapt step size.
        min_step_size: Minimum allowed step size.
        max_step_size: Maximum allowed step size.
        e_rel: Relative error tolerance.
        e_abs: Absolute error tolerance. Default assumes standardized targets.
        r: Order of the method for step size adaptation.
        adapt_safety: Safety factor for step size adaptation.

    Returns:
        new_state: Updated state after one adaptive step.
        new_time: time + dt (or time if step rejected).
        new_step_size: Adapted step size for next iteration.
    """
    state_euler, time_mid, _ = euler_maruyama_step(
        drift_fn=drift_fn,
        diffusion_fn=diffusion_fn,
        state=state,
        time=time,
        step_size=step_size,
        min_step_size=min_step_size,
        max_step_size=max_step_size,
        noise=noise,
        use_adaptive_step_size=False,
    )

    # Compute drift and diffusion at new state, but update from old state
    drift_mid = drift_fn(time_mid, **filter_kwargs(state_euler, drift_fn))
    diffusion_mid = diffusion_fn(time_mid, **filter_kwargs(state_euler, diffusion_fn))
    sqrt_step_size = keras.ops.sqrt(keras.ops.abs(step_size))

    state_euler_mid = {}
    for key, d in drift_mid.items():
        base = state[key] + step_size * d
        if key in diffusion_mid:
            base = base + diffusion_mid[key] * sqrt_step_size * noise[key]
        state_euler_mid[key] = base

    # average the two predictions
    state_heun = {}
    for key in state.keys():
        state_heun[key] = 0.5 * (state_euler[key] + state_euler_mid[key])

    # Error estimation
    if use_adaptive_step_size:
        if e_abs is None:
            e_abs = 0.02576  # 1% of 99% CI of standardized unit variance
        # Check if we're at minimum step size - if so, force acceptance
        at_min_step = keras.ops.less_equal(keras.ops.abs(step_size), min_step_size)

        # Compute error tolerance for each component
        e_abs_tensor = keras.ops.cast(e_abs, dtype=keras.ops.dtype(list(state.values())[0]))
        e_rel_tensor = keras.ops.cast(e_rel, dtype=keras.ops.dtype(list(state.values())[0]))

        max_error = keras.ops.cast(0.0, dtype=keras.ops.dtype(list(state.values())[0]))

        for key in state.keys():
            # Local error estimate: difference between Heun and first Euler step
            error_estimate = keras.ops.abs(state_heun[key] - state_euler[key])

            # Tolerance threshold
            delta = keras.ops.maximum(
                e_abs_tensor,
                e_rel_tensor * keras.ops.maximum(keras.ops.abs(state_euler[key]), keras.ops.abs(last_state[key])),
            )

            # Normalized error
            normalized_error = error_estimate / (delta + 1e-10)

            # Maximum error across all components and batch dimensions
            component_max_error = keras.ops.max(normalized_error)
            max_error = keras.ops.maximum(max_error, component_max_error)

        error_scale = 1  # 1/sqrt(n_params)
        E2 = error_scale * max_error

        # Accept step if error is acceptable OR if at minimum step size
        error_acceptable = keras.ops.less_equal(E2, keras.ops.cast(1.0, dtype=keras.ops.dtype(E2)))
        accepted = keras.ops.logical_or(error_acceptable, at_min_step)

        # Adapt step size for next iteration (only if not at minimum)
        # Ensure E2 is not zero to avoid division issues
        E2_safe = keras.ops.maximum(E2, 1e-10)

        # New step size based on error estimate
        adapt_factor = adapt_safety * keras.ops.power(E2_safe, -r)
        new_step_candidate = step_size * adapt_factor

        # Clamp to valid range
        new_step_size = keras.ops.clip(keras.ops.abs(new_step_candidate), min_step_size, max_step_size)
        new_step_size = keras.ops.sign(step_size) * new_step_size

        # Return appropriate state based on acceptance
        new_state = keras.ops.cond(accepted, lambda: state_heun, lambda: state)

        new_time = keras.ops.cond(accepted, lambda: time_mid, lambda: time)

        prev_state = keras.ops.cond(accepted, lambda: state_euler, lambda: state)

        return new_state, new_time, new_step_size, prev_state

    else:
        return state_heun, time_mid, step_size


def compute_levy_area(
    state: StateDict, diffusion: StateDict, noise: StateDict, noise_aux: StateDict, step_size: ArrayLike
) -> StateDict:
    step_size_abs = keras.ops.abs(step_size)
    sqrt_step_size = keras.ops.sqrt(step_size_abs)
    inv_sqrt3 = keras.ops.cast(1.0 / np.sqrt(3.0), dtype=keras.ops.dtype(step_size_abs))

    # Build Lévy area H_k from w_k and Z_k
    H = {}
    for k in state.keys():
        if k in diffusion:
            term1 = 0.5 * step_size_abs * noise[k]
            term2 = 0.5 * step_size_abs * sqrt_step_size * inv_sqrt3 * noise_aux[k]
            H[k] = term1 + term2
        else:
            H[k] = keras.ops.zeros_like(state[k])
    return H


def sea_step(
    drift_fn: Callable,
    diffusion_fn: Callable,
    state: StateDict,
    time: ArrayLike,
    step_size: ArrayLike,
    noise: StateDict,  # standard normals
    noise_aux: StateDict,  # standard normals
) -> Tuple[StateDict, ArrayLike, ArrayLike]:
    """
    Performs a single shifted Euler step for SDEs with additive noise [1].

    Compared to Euler-Maruyama, this evaluates the drift at a shifted state,
    which improves the local error and the global error constant for additive noise.

    The scheme is
        X_{n+1} = X_n + f(t_n, X_n + g(t_n) * (0.5 * ΔW_n + ΔH_n) * h + g(t_n) * ΔW_n

    [1] Foster et al., "High order splitting methods for SDEs satisfying a commutativity condition" (2023)
    Args:
        drift_fn: Function computing the drift term f(t, **state).
        diffusion_fn: Function computing the diffusion term g(t, **state).
        state: Current state, mapping variable names to tensors.
        time: Current time scalar tensor.
        step_size: Time increment dt.
        noise: Mapping of variable names to dW noise tensors.
        noise_aux: Mapping of variable names to auxiliary noise.

    Returns:
        new_state: Updated state after one SEA step.
        new_time: time + dt.
    """
    # Compute diffusion
    diffusion = diffusion_fn(time, **filter_kwargs(state, diffusion_fn))
    sqrt_step_size = keras.ops.sqrt(keras.ops.abs(step_size))

    la = compute_levy_area(state=state, diffusion=diffusion, noise=noise, noise_aux=noise_aux, step_size=step_size)

    # Build shifted state: X_shift = X + g * (0.5 * ΔW + ΔH)
    shifted_state = {}
    for key, x in state.items():
        if key in diffusion:
            shifted_state[key] = x + diffusion[key] * (0.5 * sqrt_step_size * noise[key] + la[key])
        else:
            shifted_state[key] = x

    # Drift evaluated at shifted state
    drift_shifted = drift_fn(time, **filter_kwargs(shifted_state, drift_fn))

    # Final update
    new_state = {}
    for key, d in drift_shifted.items():
        base = state[key] + step_size * d
        if key in diffusion:
            base = base + diffusion[key] * sqrt_step_size * noise[key]
        new_state[key] = base

    return new_state, time + step_size, step_size


def shark_step(
    drift_fn: Callable,
    diffusion_fn: Callable,
    state: StateDict,
    time: ArrayLike,
    step_size: ArrayLike,
    noise: StateDict,
    noise_aux: StateDict,
) -> Tuple[StateDict, ArrayLike, ArrayLike]:
    """
    Shifted Additive noise Runge Kutta (SHARK) for additive SDEs [1]. Makes two evaluations of the drift and diffusion
    per step and has a strong order 1.5.

    SHARK method as specified:

        1)  ỹ_k = y_k + g(y_k) H_k
        2)  ỹ_{k+5/6} = ỹ_k + (5/6)[ f(ỹ_k) h + g(ỹ_k) W_k ]
        3)  y_{k+1} = y_k
                     + (2/5) f(ỹ_k) h
                     + (3/5) f(ỹ_{k+5/6}) h
                     + g(ỹ_k) ( 2/5 W_k +  6/5 H_k )
                     + g(ỹ_{k+5/6}) ( 3/5 W_k -  6/5 H_k )

        with
            H_k = 0.5 * |h| * W_k + (|h| ** 1.5) / (2 * sqrt(3)) * Z_k

    [1] Foster et al., "High order splitting methods for SDEs satisfying a commutativity condition" (2023)

    Args:
        drift_fn: Function computing the drift term f(t, **state).
        diffusion_fn: Function computing the diffusion term g(t, **state).
        state: Current state, mapping variable names to tensors.
        time: Current time scalar tensor.
        step_size: Time increment dt.
        noise: Mapping of variable names to dW noise tensors.
        noise_aux: Mapping of variable names to auxiliary noise.

    Returns:
        new_state: Updated state after one SHARK step.
        new_time: time + dt.
    """
    h = step_size
    t = time
    h_mag = keras.ops.abs(h)
    sqrt_h_mag = keras.ops.sqrt(h_mag)

    diffusion = diffusion_fn(t, **filter_kwargs(state, diffusion_fn))

    la = compute_levy_area(state=state, diffusion=diffusion, noise=noise, noise_aux=noise_aux, step_size=step_size)

    # === 1) shifted initial state ===
    y_tilde_k = {}
    for k in state.keys():
        if k in diffusion:
            y_tilde_k[k] = state[k] + diffusion[k] * la[k]
        else:
            y_tilde_k[k] = state[k]

    # === evaluate drift and diffusion at ỹ_k ===
    f_tilde_k = drift_fn(t, **filter_kwargs(y_tilde_k, drift_fn))
    g_tilde_k = diffusion_fn(t, **filter_kwargs(y_tilde_k, diffusion_fn))

    # === 2) internal stage at 5/6 ===
    y_tilde_mid = {}
    for k in state.keys():
        drift_part = (5.0 / 6.0) * f_tilde_k[k] * h
        if k in g_tilde_k:
            sto_part = (5.0 / 6.0) * g_tilde_k[k] * sqrt_h_mag * noise[k]
        else:
            sto_part = keras.ops.zeros_like(state[k])
        y_tilde_mid[k] = y_tilde_k[k] + drift_part + sto_part

    # === evaluate drift and diffusion at ỹ_(k+5/6) ===
    f_tilde_mid = drift_fn(t + 5.0 / 6.0 * h, **filter_kwargs(y_tilde_mid, drift_fn))
    g_tilde_mid = diffusion_fn(t + 5.0 / 6.0 * h, **filter_kwargs(y_tilde_mid, diffusion_fn))

    # === 3) final update ===
    new_state = {}
    for k in state.keys():
        # deterministic weights
        det = state[k] + (2.0 / 5.0) * f_tilde_k[k] * h + (3.0 / 5.0) * f_tilde_mid[k] * h

        # stochastic parts
        sto1 = (
            g_tilde_k[k] * ((2.0 / 5.0) * sqrt_h_mag * noise[k] + (6.0 / 5.0) * la[k])
            if k in g_tilde_k
            else keras.ops.zeros_like(det)
        )
        sto2 = (
            g_tilde_mid[k] * ((3.0 / 5.0) * sqrt_h_mag * noise[k] - (6.0 / 5.0) * la[k])
            if k in g_tilde_mid
            else keras.ops.zeros_like(det)
        )

        new_state[k] = det + sto1 + sto2

    return new_state, t + h, h


def _apply_corrector(
    new_state: StateDict,
    new_time: ArrayLike,
    i: ArrayLike,
    corrector_steps: int,
    score_fn: Optional[Callable],
    corrector_noise_history: StateDict | None,
    seed: keras.random.SeedGenerator,
    step_size_factor: ArrayLike = 0.01,
    noise_schedule=None,
) -> StateDict:
    """Helper function to apply corrector steps [1].

    [1] Song et al., "Score-Based Generative Modeling through Stochastic Differential Equations" (2020)
    """
    for j in range(corrector_steps):
        score = score_fn(new_time, **filter_kwargs(new_state, score_fn))
        if corrector_noise_history is None:
            _z_corr = generate_noise(new_state, seed=seed)
        else:
            _z_corr = {k: val[i, j] for k, val in corrector_noise_history.items()}

        log_snr_t = noise_schedule.get_log_snr(t=new_time, training=False)
        alpha_t, _ = noise_schedule.get_alpha_sigma(log_snr_t=log_snr_t)

        for k in new_state.keys():
            if k in score:
                # Calculate required norms for Langevin step
                z_norm = keras.ops.norm(_z_corr[k], axis=-1, keepdims=True)
                score_norm = keras.ops.norm(score[k], axis=-1, keepdims=True)
                score_norm = keras.ops.maximum(score_norm, 1e-8)

                # Compute step size for the Langevin update
                e = 2.0 * alpha_t * (step_size_factor * z_norm / score_norm) ** 2

                # Annealed Langevin Dynamics update
                new_state[k] = new_state[k] + e * score[k] + keras.ops.sqrt(2.0 * e) * _z_corr[k]
    return new_state


def integrate_stochastic_fixed(
    step_fn: Callable,
    state: StateDict,
    start_time: ArrayLike,
    stop_time: ArrayLike,
    steps: int,
    min_step_size: ArrayLike,
    max_step_size: ArrayLike,
    z_history: StateDict | None,
    z_extra_history: StateDict | None,
    score_fn: Optional[Callable],
    step_size_factor: ArrayLike,
    corrector_noise_history: StateDict | None,
    seed: keras.random.SeedGenerator,
    corrector_steps: int = 0,
    noise_schedule=None,
) -> StateDict:
    """
    Performs fixed-step SDE integration.
    """
    initial_step = (stop_time - start_time) / float(steps)

    def cond(_loop_var, _loop_state, _loop_time, _loop_step):
        all_nans = _check_all_nans(_loop_state)
        end_now = keras.ops.less(_loop_var, steps)
        return keras.ops.logical_and(~all_nans, end_now)

    def body(_i, _current_state, _current_time, _current_step):
        # Determine step size: either the constant size or the remainder to reach stop_time
        remaining = keras.ops.abs(stop_time - _current_time)
        sign = keras.ops.sign(_current_step)
        dt_mag = keras.ops.minimum(keras.ops.abs(_current_step), remaining)
        dt = sign * dt_mag

        # Generate noise increment
        if z_history is None:
            _noise_i = generate_noise(_current_state, seed=seed)
        else:
            _noise_i = {k: val[_i] for k, val in z_history.items()}
        _noise_extra_i = None
        if z_extra_history is not None:
            if len(z_extra_history) == 0:
                _noise_extra_i = generate_noise(_current_state, seed=seed)
            else:
                _noise_extra_i = {k: val[_i] for k, val in z_history.items()}

        step_fn_additional_args = dict(
            noise_aux=_noise_extra_i,
            min_step_size=min_step_size,
            max_step_size=keras.ops.minimum(max_step_size, remaining),
            use_adaptive_step_size=False,
        )
        new_state, new_time, new_step = step_fn(
            state=_current_state,
            time=_current_time,
            step_size=dt,
            noise=_noise_i,
            **filter_kwargs(step_fn_additional_args, step_fn),
        )

        if corrector_steps > 0:
            new_state = _apply_corrector(
                new_state=new_state,
                new_time=new_time,
                i=_i,
                corrector_steps=corrector_steps,
                score_fn=score_fn,
                noise_schedule=noise_schedule,
                step_size_factor=step_size_factor,
                corrector_noise_history=corrector_noise_history,
                seed=seed,
            )
        return _i + 1, new_state, new_time, initial_step

    _, final_state, final_time, _ = keras.ops.while_loop(
        cond,
        body,
        [0, state, start_time, initial_step],
    )
    # skipped while tracing
    if _as_concrete_bool(_check_all_nans(final_state)):
        raise RuntimeError(f"All values are NaNs in state during integration at {final_time}.")

    return final_state


def integrate_stochastic_adaptive(
    step_fn: Callable,
    state: StateDict,
    start_time: ArrayLike,
    stop_time: ArrayLike,
    max_steps: int,
    min_step_size: ArrayLike,
    max_step_size: ArrayLike,
    initial_step: ArrayLike,
    z_history: StateDict | None,
    z_extra_history: StateDict | None,
    score_fn: Optional[Callable],
    step_size_factor: ArrayLike,
    seed: keras.random.SeedGenerator,
    corrector_noise_history: StateDict | None,
    corrector_steps: int = 0,
    noise_schedule=None,
) -> StateDict:
    """
    Performs adaptive-step SDE integration.
    """
    initial_loop_state = (keras.ops.zeros((), dtype="int32"), state, start_time, initial_step, state)
    if keras.backend.backend() == "jax":
        seed = None  # not needed, noise is generated upfront
    seed_body = seed

    def cond(i, current_state, current_time, current_step, last_state):
        time_remaining = keras.ops.sign(stop_time - start_time) * (stop_time - (current_time + current_step))
        all_nans = _check_all_nans(current_state)
        end_now = keras.ops.logical_and(keras.ops.all(time_remaining > 0), keras.ops.less(i, max_steps))
        return keras.ops.logical_and(~all_nans, end_now)

    def body_adaptive(_i, _current_state, _current_time, _current_step, _last_state):
        # Step Size Control
        remaining = keras.ops.abs(stop_time - _current_time)
        sign = keras.ops.sign(_current_step)
        # Ensure the next step does not overshoot the stop_time
        dt_mag = keras.ops.minimum(keras.ops.abs(_current_step), remaining)
        dt = sign * dt_mag

        if z_history is None:
            _noise_i = generate_noise(_current_state, seed=seed_body)
        else:
            _noise_i = {k: val[_i] for k, val in z_history.items()}

        _noise_extra_i = None
        if z_extra_history is not None:
            if len(z_extra_history) == 0:
                _noise_extra_i = generate_noise(_current_state, seed=seed_body)
            else:
                _noise_extra_i = {k: val[_i] for k, val in z_history.items()}

        step_fn_additional_args = dict(
            last_state=_last_state,
            noise_aux=_noise_extra_i,
        )
        new_state, new_time, new_step, _new_current_state = step_fn(
            state=_current_state,
            time=_current_time,
            step_size=dt,
            min_step_size=min_step_size,
            max_step_size=keras.ops.minimum(max_step_size, remaining),
            noise=_noise_i,
            use_adaptive_step_size=True,
            **filter_kwargs(step_fn_additional_args, step_fn),
        )

        if corrector_steps > 0:
            new_state = _apply_corrector(
                new_state=new_state,
                new_time=new_time,
                i=_i,
                corrector_steps=corrector_steps,
                score_fn=score_fn,
                noise_schedule=noise_schedule,
                step_size_factor=step_size_factor,
                corrector_noise_history=corrector_noise_history,
                seed=seed_body,
            )

        return _i + 1, new_state, new_time, new_step, _new_current_state

    # Execute the adaptive loop
    final_counter, final_state, final_time, _, final_k1 = keras.ops.while_loop(cond, body_adaptive, initial_loop_state)

    # skipped while tracing
    if _as_concrete_bool(_check_all_nans(final_state)):
        raise RuntimeError(f"All values are NaNs in state during integration at {final_time}.")

    # Final step to hit stop_time exactly
    time_diff = stop_time - final_time
    time_remaining = keras.ops.sign(stop_time - start_time) * time_diff

    def take_final_step():
        noise_final = generate_noise(final_state, seed=seed)
        noise_extra_final = None
        if z_extra_history is not None and len(z_extra_history) > 0:
            noise_extra_final = generate_noise(final_state, seed=seed)

        step_fn_additional_args_final = dict(
            noise_aux=noise_extra_final,
            last_state=final_k1,
        )
        stepped_state, _, _ = step_fn(
            state=final_state,
            time=final_time,
            step_size=time_diff,
            min_step_size=min_step_size,
            max_step_size=time_remaining,
            noise=noise_final,
            use_adaptive_step_size=False,
            **filter_kwargs(step_fn_additional_args_final, step_fn),
        )
        return stepped_state, final_counter + 1

    take_final = _as_concrete_bool(keras.ops.all(time_remaining > 0))
    if take_final is None:
        # tracing inside tf.function: express the branch in-graph. On eager backends the Python
        # branch below is kept instead, since jax.lax.cond would trace the noise generation
        final_state, final_counter = keras.ops.cond(
            keras.ops.all(time_remaining > 0), take_final_step, lambda: (final_state, final_counter)
        )
    elif take_final:
        final_state, final_counter = take_final_step()

    if take_final is not None:  # not tracing, values are concrete
        debug(f"Finished integration after {final_counter}.")
    return final_state


def integrate_langevin(
    state: StateDict,
    start_time: ArrayLike,
    stop_time: ArrayLike,
    steps: int,
    z_history: StateDict | None,
    score_fn: Callable,
    noise_schedule,
    seed: keras.random.SeedGenerator,
    corrector_noise_history: StateDict | None,
    step_size_factor: ArrayLike = 0.01,
    corrector_steps: int = 0,
) -> StateDict:
    """
    Annealed Langevin dynamics using the given score_fn and noise_schedule [1].

    At each step i with time t_i, performs for every state component k:
        state_k <- state_k + e * score_k + sqrt(2 * e) * z

    Times are stepped linearly from start_time to stop_time.

    [1]  Song et al., "Generative Modeling by Estimating Gradients of the Data Distribution" (2020)
    """

    if steps <= 0:
        raise ValueError("Number of Langevin steps must be positive.")
    if score_fn is None or noise_schedule is None:
        raise ValueError("score_fn and noise_schedule must be provided.")

    # Linear time grid
    dt = (stop_time - start_time) / float(steps)
    effective_factor = step_size_factor * 100 / np.sqrt(steps)

    def cond(_loop_var, _loop_state, _loop_time):
        all_nans = _check_all_nans(_loop_state)
        end_now = keras.ops.less(_loop_var, steps)
        return keras.ops.logical_and(~all_nans, end_now)

    def body(_i, _loop_state, _loop_time):
        # score at current time
        score = score_fn(_loop_time, **filter_kwargs(_loop_state, score_fn))

        # noise schedule
        log_snr_t = noise_schedule.get_log_snr(t=_loop_time, training=False)
        _, sigma_t = noise_schedule.get_alpha_sigma(log_snr_t=log_snr_t)

        new_state: StateDict = {}
        if z_history is None:
            z_history_i = generate_noise(_loop_state, seed=seed)
        else:
            z_history_i = {k: val[_i] for k, val in z_history.items()}
        for k in _loop_state.keys():
            s_k = score.get(k, None)
            if s_k is None:
                new_state[k] = _loop_state[k]
                continue

            e = effective_factor * sigma_t**2
            new_state[k] = _loop_state[k] + e * s_k + keras.ops.sqrt(2.0 * e) * z_history_i[k]

        new_time = _loop_time + dt

        if corrector_steps > 0:
            new_state = _apply_corrector(
                new_state=new_state,
                new_time=new_time,
                i=_i,
                corrector_steps=corrector_steps,
                score_fn=score_fn,
                noise_schedule=noise_schedule,
                step_size_factor=step_size_factor,
                corrector_noise_history=corrector_noise_history,
                seed=seed,
            )

        return _i + 1, new_state, new_time

    _, final_state, final_time = keras.ops.while_loop(
        cond,
        body,
        (0, state, start_time),
    )
    # skipped while tracing
    if _as_concrete_bool(_check_all_nans(final_state)):
        raise RuntimeError(f"All values are NaNs in state during integration at {final_time}.")
    return final_state


def _glass_alpha_sigma(noise_schedule, t: Tensor) -> Tuple[Tensor, Tensor]:
    """Return (alpha_t, sigma_t) for the given schedule at time t.

    Accepts either the string ``"flow_matching"`` (alpha_t = t, sigma_t = 1 - t)
    or any ``NoiseSchedule`` instance.
    """
    if noise_schedule == "flow_matching":
        return t, 1.0 - t
    log_snr = noise_schedule.get_log_snr(t, training=False)
    return noise_schedule.get_alpha_sigma(log_snr)


def _glass_g_inv(noise_schedule, v: Tensor) -> Tensor:
    """
    Invert g(t) = sigma_t^2 / alpha_t^2 to obtain t* s.t. g(t*) = v.

    For implemented diffusion noise schedule g(t) = exp(-lambda_t).
    """
    if noise_schedule == "flow_matching":
        t_star = 1.0 / (1.0 + keras.ops.sqrt(v))
    else:
        # g(t) = exp(-lamda_t)
        log_snr_star = -keras.ops.log(v)
        t_star = noise_schedule.get_t_from_log_snr(log_snr_star, training=False)
    return t_star


def _glass_alpha_sigma_dot(noise_schedule, t: Tensor) -> Tuple[Tensor, Tensor]:
    """Return (alpha_dot_t, sigma_dot_t) — analytical schedule derivatives at t.

    Uses the identity beta(t) = d/dt log(1 + e^{-lambda(t)}) = -lambda_dot * sigmoid(-lambda),
    plus the variance-preserving / variance-exploding parameterizations of (alpha, sigma)
    in terms of lambda.

    Flow matching:        alpha_t = t,  sigma_t = 1 - t   (closed form)
    Variance preserving:  alpha_dot = -0.5 * alpha * beta
                          sigma_dot =  0.5 * alpha^2 * beta / sigma
    Variance exploding:   alpha_dot = 0
                          sigma_dot = beta * (1 + sigma^2) / (2 * sigma)
    """
    if noise_schedule == "flow_matching":
        ones = keras.ops.ones_like(t)
        return ones, -ones

    log_snr = noise_schedule.get_log_snr(t, training=False)
    alpha, sigma = noise_schedule.get_alpha_sigma(log_snr)
    beta = noise_schedule.derivative_log_snr(log_snr, training=False)

    if noise_schedule._variance_type == "preserving":
        alpha_dot = -0.5 * alpha * beta
        sigma_dot = 0.5 * alpha**2 * beta / sigma
    elif noise_schedule._variance_type == "exploding":
        alpha_dot = keras.ops.zeros_like(beta)
        sigma_dot = beta * (1.0 + sigma**2) / (2.0 * sigma)
    else:
        raise ValueError(f"Unknown variance type for GLASS: {noise_schedule._variance_type!r}")
    return alpha_dot, sigma_dot


def _glass_suf_stat(
    x_t: Tensor,
    x_bar_s: Tensor,
    alpha_t: Tensor,
    sigma_t: Tensor,
    alpha_bar_s: Tensor,
    sigma_bar_s: Tensor,
    gamma_bar: Tensor,
) -> Tuple[Tensor, Tensor]:
    """
    Compute the GLASS sufficient statistic S.

    For the structured covariance

        Sigma = [[sigma_t²,               sigma_t² * gamma_bar           ],
                 [sigma_t² * gamma_bar,   sigma_bar_s² + gamma_bar² * sigma_t²]]

    det = sigma_t² * sigma_bar_s², giving closed-form weights::

        w_t  = alpha_t / sigma_t²  -  alpha_bar_s * gamma_bar / sigma_bar_s²
        w_b  = alpha_bar_s / sigma_bar_s²
        norm = alpha_t² / sigma_t²  +  alpha_bar_s² / sigma_bar_s²
        S    = (w_t * x_t + w_b * x_bar_s) / norm

    Returns ``(S, norm)``.
    """
    sigma_t_sq = sigma_t**2
    sigma_bar_s_sq = sigma_bar_s**2

    w_t = alpha_t / sigma_t_sq - alpha_bar_s * gamma_bar / sigma_bar_s_sq
    w_b = alpha_bar_s / sigma_bar_s_sq
    norm = alpha_t**2 / sigma_t_sq + alpha_bar_s**2 / sigma_bar_s_sq
    norm = keras.ops.maximum(norm, keras.ops.cast(1e-12, keras.ops.dtype(norm)))
    suf_stat = (w_t * x_t + w_b * x_bar_s) / norm
    return suf_stat, norm


def glass_step(
    fn: Callable,
    state: StateDict,
    time: ArrayLike,
    step_size: ArrayLike,
    noise_schedule,
    *,
    x_t: StateDict,
    alpha_t,
    sigma_t,
    alpha_bar,
    sigma_bar,
    gamma_bar,
    sigma_bar_0: float,
) -> Tuple[StateDict, ArrayLike]:
    """
    Single Euler step of the GLASS inner ODE at inner time s in [0, 1).

    ``fn`` returns the velocity field; denoiser is derived using the noise schedule ``noise_schedule``.
    """
    dtype = keras.ops.dtype(next(iter(x_t.values())))
    s_val = keras.ops.cast(time, dtype)

    alpha_bar_s = s_val * alpha_bar
    sigma_bar_s = (1 - s_val) * sigma_bar_0 + s_val * sigma_bar
    dot_alpha_bar_s = alpha_bar
    dot_sigma_bar_s = sigma_bar - sigma_bar_0

    w1 = dot_sigma_bar_s / sigma_bar_s
    w2 = dot_alpha_bar_s - alpha_bar_s * w1
    w3 = -gamma_bar * w1

    suf_stats = {}
    norm = None
    for key, x_bar_s in state.items():
        suf_stat, norm = _glass_suf_stat(x_t[key], x_bar_s, alpha_t, sigma_t, alpha_bar_s, sigma_bar_s, gamma_bar)
        suf_stats[key] = suf_stat

    t_star = _glass_g_inv(noise_schedule, 1.0 / norm)
    alpha_t_star, sigma_t_star = _glass_alpha_sigma(noise_schedule, t_star)
    dot_alpha_t_star, dot_sigma_t_star = _glass_alpha_sigma_dot(noise_schedule, t_star)
    denom = dot_alpha_t_star * sigma_t_star - alpha_t_star * dot_sigma_t_star

    z_in = {key: alpha_t_star * suf_stat for key, suf_stat in suf_stats.items()}
    fn_out = fn(t_star, **filter_kwargs(z_in, fn))
    missing_keys = set(state) - set(fn_out)
    if missing_keys:
        raise ValueError(f"GLASS model output is missing state keys: {sorted(missing_keys)}.")

    new_state = {}
    for key, x_bar_s in state.items():
        d_val = (sigma_t_star * fn_out[key] - dot_sigma_t_star * z_in[key]) / denom

        velocity = w1 * x_bar_s + w2 * d_val + w3 * x_t[key]
        new_state[key] = x_bar_s + step_size * velocity

    return new_state, time + step_size


def integrate_glass(
    fn: Callable,
    state: StateDict,
    start_time: ArrayLike,
    stop_time: ArrayLike,
    steps: int,
    seed: int | keras.random.SeedGenerator | None,
    noise_schedule,
    inner_steps: int = 1,
    rho: float = 0.4,
    sigma_bar_0: float = 1.0,
    **kwargs,
) -> StateDict:
    """
    Sample from the Markov transition p_{t'|t}(·|x_t) using the GLASS inner ODE [1].

    Constructs a deterministic inner ODE whose endpoint approximates a draw
    from p_{t'|t}(·|x_t) implied by the trained model. Allows stochastic sampling from a flow matching model.

    [1] Holderrieth et al. (2026) "GLASS Flows: Transition Sampling for Alignment of Flow and Diffusion Models"

    Parameters
    ----------
    fn : Callable
        Model forward function in integrate-compatible form. It receives
        ``fn(time, **state)`` velocity field and returns a state dictionary with matching keys.
    state : dict
        Current trajectory state.
    start_time : float or Tensor
        Start outer time.
    stop_time : float or Tensor
        Target outer time.
    steps : int
        Number of GLASS transitions. K=1 standard flow matching integration.
    noise_schedule : "flow_matching" or NoiseSchedule, optional
        Outer noise schedule.  Pass "flow_matching" for flow matching; pass the model's
        ".noise_schedule" attribute for diffusion models.
        Note that flow matching and diffusion models operate in different time directions.
    seed : int, SeedGenerator, or None
        Seed for the stochastic inner initialization.
    inner_steps : int
        Inner GLASS-ODE steps M. M=1 recovers DDIM.
    rho : float, optional
        Correlation ρ ∈ [-1, 1]. Default 0.4.
    sigma_bar_0 : float, optional
        Initial inner noise level (> 0).  Default 1.
    Returns
    -------
    dict
        State dictionary containing the approximate transition sample.

    """
    if not isinstance(steps, int) or steps <= 0:
        raise ValueError(f"GLASS integration requires a positive integer number of steps, got {steps}.")
    if start_time is None or stop_time is None:
        raise ValueError(
            "Please provide start_time and stop_time for GLASS integration, was "
            f"'start_time={start_time}', 'stop_time={stop_time}'."
        )
    if not state:
        raise ValueError("GLASS integration requires a non-empty state dictionary.")
    if sigma_bar_0 < 0:
        raise ValueError(f"sigma_bar_0 must be > 0, got {sigma_bar_0}.")

    x_t = {key: keras.ops.convert_to_tensor(value) for key, value in state.items()}
    dtype = keras.ops.dtype(next(iter(x_t.values())))
    t_val = keras.ops.cast(start_time, dtype)
    t_prime_val = keras.ops.cast(stop_time, dtype)
    rho_val = keras.ops.cast(rho, dtype)
    sigma_bar_0_val = keras.ops.cast(sigma_bar_0, dtype)

    glass_transitions = steps  # K
    outer_step_size = (t_prime_val - t_val) / keras.ops.cast(glass_transitions, dtype)  # K
    inner_step_size = keras.ops.cast(1.0 / inner_steps, dtype)

    # Pre-generate noise for all outer steps
    noise_per_step = {}
    for key, value in x_t.items():
        noise_per_step[key] = keras.random.normal(
            (glass_transitions, *keras.ops.shape(value)),
            dtype=keras.ops.dtype(value),
            seed=seed,
        )

    def outer_body(i, current_state):
        i_f = keras.ops.cast(i, dtype)
        t_curr = t_val + i_f * outer_step_size
        t_next = t_val + (i_f + 1.0) * outer_step_size

        alpha_curr, sigma_curr = _glass_alpha_sigma(noise_schedule, t_curr)
        alpha_next, sigma_next = _glass_alpha_sigma(noise_schedule, t_next)

        # Derived scalars for this sub-interval
        gamma_bar_i = rho_val * sigma_next / sigma_curr
        alpha_bar_i = alpha_next - gamma_bar_i * alpha_curr
        sigma_bar_i = keras.ops.sqrt(sigma_next**2 * (1.0 - rho_val**2))

        inner_state_i = {
            key: gamma_bar_i * current_state[key] + sigma_bar_0_val * noise_per_step[key][i] for key in current_state
        }

        def inner_body(_j, _inner_loop_state):
            _state, _time = _inner_loop_state
            _state, _time = glass_step(
                fn,
                _state,
                _time,
                inner_step_size,
                noise_schedule=noise_schedule,
                x_t=current_state,
                alpha_t=alpha_curr,
                sigma_t=sigma_curr,
                alpha_bar=alpha_bar_i,
                sigma_bar=sigma_bar_i,
                gamma_bar=gamma_bar_i,
                sigma_bar_0=sigma_bar_0_val,
            )
            return _state, _time

        inner_state_i, _ = keras.ops.fori_loop(0, inner_steps, inner_body, (inner_state_i, keras.ops.cast(0.0, dtype)))
        return inner_state_i

    return keras.ops.fori_loop(0, glass_transitions, outer_body, x_t)


def integrate_stochastic(
    drift_fn: Callable,
    diffusion_fn: Callable,
    state: StateDict,
    start_time: ArrayLike,
    stop_time: ArrayLike,
    seed: keras.random.SeedGenerator,
    steps: int | Literal["adaptive"] = 100,
    method: str = "euler_maruyama",
    min_steps: int = 50,
    max_steps: int = 1_000,
    score_fn: Callable = None,
    corrector_steps: int = 0,
    noise_schedule=None,
    step_size_factor: ArrayLike = 0.01,
    **kwargs,
) -> StateDict:
    """
    Integrate a stochastic differential equation from ``start_time`` to ``stop_time``.

    This function dispatches to either fixed-step or adaptive-step integration logic,
    depending on the selected integration ``method`` and the value of ``steps``.

    Parameters
    ----------
    drift_fn : Callable
        Function computing the drift term of the SDE. It should accept the current
        state and time as inputs.
    diffusion_fn : Callable
        Function computing the diffusion term of the SDE. It should accept the current
        state and time as inputs.
    state : StateDict
        Dictionary containing the initial state of the system.
    start_time : array-like
        Starting time for integration.
    stop_time : array-like
        Ending time for integration.
    seed : keras.random.SeedGenerator
        Random seed generator used for noise generation.
    steps : int or {'adaptive'}, optional
        Number of integration steps for fixed-step integration, or ``'adaptive'``
        to enable adaptive step sizing. Adaptive integration is only supported by
        the ``'shark'`` method. Default is ``100``.
    method : str, optional
        Integration method to use (e.g., ``'euler_maruyama'`` or ``'shark'``).
        Default is ``'euler_maruyama'``.
    min_steps : int, optional
        Minimum number of steps for adaptive integration. Default is ``50``.
    max_steps : int, optional
        Maximum number of steps for adaptive integration. Noise is pre-generated
        up to this number of steps, which may increase memory usage. Default is
        ``1000``.
    score_fn : Callable, optional
        Score function used for predictor-corrector sampling. If ``None``,
        no corrector step is applied.
    corrector_steps : int, optional
        Number of corrector steps applied after each predictor step. Default is ``0``.
    noise_schedule : object, optional
        Noise schedule object used to compute ``alpha_t`` during the corrector step.
        Required if ``corrector_steps > 0``.
    step_size_factor : array-like, optional
        Scaling factor applied to the corrector step size. Default is ``0.01``.
    **kwargs
        Additional keyword arguments passed to the underlying step function.

    Returns
    -------
    StateDict
        Final state dictionary after integration.
    """

    if isinstance(steps, str) and steps not in ["adaptive", "dynamic"]:
        raise ValueError(f"Unknown steps value: {steps!r}. Use an integer or 'adaptive'.")
    is_adaptive = isinstance(steps, str) and steps in ["adaptive", "dynamic"]

    if method == "glass":
        if is_adaptive:
            raise ValueError("Adaptive step sizing is not supported for the 'glass' method.")
        if isinstance(steps, Sequence) or isinstance(steps, np.ndarray) or keras.ops.is_tensor(steps):
            raise ValueError("Scheduled integration is not supported for the 'glass' method.")

        def run_glass():
            return integrate_glass(
                fn=drift_fn,
                state=state,
                start_time=start_time,
                stop_time=stop_time,
                steps=steps,
                noise_schedule=noise_schedule or "flow_matching",
                seed=seed,
                **kwargs,
            )

        return _compile_loop_integrator(run_glass)()

    if is_adaptive:
        if start_time is None or stop_time is None:
            raise ValueError("Please provide start_time and stop_time for adaptive integration.")
        if min_steps <= 0 or max_steps <= 0 or max_steps < min_steps:
            raise ValueError("min_steps and max_steps must be positive, and max_steps >= min_steps.")

        loop_steps = max_steps
        initial_step = (stop_time - start_time) / float(min_steps)
        span_mag = keras.ops.abs(stop_time - start_time)
        min_step_size = span_mag / keras.ops.cast(max_steps, span_mag.dtype)
        max_step_size = span_mag / keras.ops.cast(min_steps, span_mag.dtype)
    else:
        if steps <= 0:
            raise ValueError("Number of steps must be positive.")
        loop_steps = int(steps)
        initial_step = (stop_time - start_time) / float(loop_steps)
        # For fixed step, min/max step size are just the fixed step size
        min_step_size, max_step_size = initial_step, initial_step

    # Pre-generate corrector noise if requested
    corrector_noise_history = None
    if corrector_steps > 0:
        if score_fn is None or noise_schedule is None:
            raise ValueError("Please provide both score_fn and noise_schedule when using corrector_steps > 0.")
        if keras.backend.backend() == "jax":
            corrector_noise_history = {}
            for key, val in state.items():
                shape = keras.ops.shape(val)
                corrector_noise_history[key] = keras.random.normal(
                    (loop_steps, corrector_steps, *shape), dtype=keras.ops.dtype(val), seed=seed
                )

    match method:
        case "euler_maruyama":
            step_fn_raw = euler_maruyama_step
        case "sea":
            step_fn_raw = sea_step
            if is_adaptive:
                raise ValueError("SEA SDE solver does not support adaptive steps.")
        case "shark":
            step_fn_raw = shark_step
            if is_adaptive:
                raise ValueError("SHARK SDE solver does not support adaptive steps.")
        case "two_step_adaptive":
            step_fn_raw = two_step_adaptive_step
        case "langevin":
            if is_adaptive:
                raise ValueError("Langevin sampling does not support adaptive steps.")

            z_history = None
            if keras.backend.backend() == "jax":
                warning(f"JAX backend needs to preallocate random samples for 'max_steps={loop_steps}'.")
                z_history = {}
                for key, val in state.items():
                    shape = keras.ops.shape(val)
                    z_history[key] = keras.random.normal((loop_steps, *shape), dtype=keras.ops.dtype(val), seed=seed)

            def run_langevin():
                return integrate_langevin(
                    state=state,
                    start_time=start_time,
                    stop_time=stop_time,
                    steps=loop_steps,
                    z_history=z_history,
                    score_fn=score_fn,
                    noise_schedule=noise_schedule,
                    step_size_factor=step_size_factor,
                    corrector_steps=corrector_steps,
                    corrector_noise_history=corrector_noise_history,
                    seed=seed,
                )

            result = _compile_loop_integrator(run_langevin)()
            # covers the TensorFlow backend, where the check inside integrate_langevin is skipped while tracing
            if _as_concrete_bool(_check_all_nans(result)):
                raise RuntimeError("All values are NaNs in state during integration.")
            return result
        case other:
            raise TypeError(f"Invalid integration method: {other!r}")

    # Partial the step function with common arguments
    step_fn = partial(step_fn_raw, drift_fn=drift_fn, diffusion_fn=diffusion_fn, **filter_kwargs(kwargs, step_fn_raw))

    # Pre-generate standard normals for the predictor step (up to max_steps)
    z_history = None
    z_extra_history = None if method not in ["sea", "shark"] else {}
    if keras.backend.backend() == "jax":
        warning(f"JAX backend needs to preallocate random samples for 'max_steps={loop_steps}'.")
        z_history = {}
        for key, val in state.items():
            shape = keras.ops.shape(val)
            z_history[key] = keras.random.normal((loop_steps, *shape), dtype=keras.ops.dtype(val), seed=seed)
            if method in ["sea", "shark"]:
                z_extra_history[key] = keras.random.normal((loop_steps, *shape), dtype=keras.ops.dtype(val), seed=seed)

    def run():
        if is_adaptive:
            return integrate_stochastic_adaptive(
                step_fn=step_fn,
                state=state,
                start_time=start_time,
                stop_time=stop_time,
                max_steps=max_steps,
                min_step_size=min_step_size,
                max_step_size=max_step_size,
                initial_step=initial_step,
                z_history=z_history,
                z_extra_history=z_extra_history,
                corrector_steps=corrector_steps,
                score_fn=score_fn,
                noise_schedule=noise_schedule,
                step_size_factor=step_size_factor,
                corrector_noise_history=corrector_noise_history,
                seed=seed,
            )
        else:
            return integrate_stochastic_fixed(
                step_fn=step_fn,
                state=state,
                start_time=start_time,
                stop_time=stop_time,
                min_step_size=min_step_size,
                max_step_size=max_step_size,
                steps=loop_steps,
                z_history=z_history,
                z_extra_history=z_extra_history,
                corrector_steps=corrector_steps,
                score_fn=score_fn,
                noise_schedule=noise_schedule,
                step_size_factor=step_size_factor,
                corrector_noise_history=corrector_noise_history,
                seed=seed,
            )

    result = _compile_loop_integrator(run)()
    # covers the TensorFlow backend, where the checks inside the integrators are skipped while tracing
    if _as_concrete_bool(_check_all_nans(result)):
        raise RuntimeError("All values are NaNs in state during integration.")
    return result
