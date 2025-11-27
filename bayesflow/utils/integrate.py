from collections.abc import Callable, Sequence
from typing import Dict, Tuple, Optional
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
StateDict = Dict[str, ArrayLike]


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
    if use_adaptive_step_size:
        raise ValueError("Adaptive step size not supported for Euler method.")

    k1 = fn(time, **filter_kwargs(state, fn))

    new_state = state.copy()
    for key in k1.keys():
        new_state[key] = state[key] + step_size * k1[key]
    new_time = time + step_size

    return new_state, new_time, step_size


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
            state, [k1, k2, k3, k4], [5.325864828439257, -11.74888356406283, 7.495539342889836, -0.09249506636175525], h
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
        k7 = fn(time + h, **filter_kwargs(new_state, fn))

        err_state = {}
        for key in state.keys():
            err_state[key] = h * (
                -0.00178001105222577714 * k1[key]
                - 0.0008164344596567469 * k2[key]
                + 0.007880878010261995 * k3[key]
                - 0.1447110071732629 * k4[key]
                + 0.5823571654525552 * k5[key]
                - 0.45808210592918697 * k6[key]
                + 0.015151515151515152 * k7[key]
            )

        err_norm = keras.ops.stack([keras.ops.norm(v, ord=2, axis=-1) for v in err_state.values()])
        err = keras.ops.max(err_norm)

        new_step_size = h * keras.ops.clip(0.9 * (tolerance / (err + 1e-12)) ** 0.2, 0.2, 5.0)
        new_step_size = keras.ops.clip(new_step_size, min_step_size, max_step_size)
    else:
        new_step_size = step_size

    new_time = time + h
    return new_state, new_time, new_step_size


def integrate_fixed(
    fn: Callable,
    state: dict[str, ArrayLike],
    start_time: ArrayLike,
    stop_time: ArrayLike,
    steps: int,
    method: str,
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
    min_steps: int,
    max_steps: int,
    method: str,
    **kwargs,
) -> dict[str, ArrayLike]:
    if max_steps <= min_steps:
        raise ValueError("Maximum number of steps must be greater than minimum number of steps.")

    match method:
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
    method: str,
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
    method: str = "rk45",
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
    noise_aux: dict[str, ArrayLike] = None,
    use_adaptive_step_size: bool = False,
    min_step_size: ArrayLike = None,
    max_step_size: ArrayLike = None,
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
        noise_aux: Mapping of variable names to auxiliary noise (not used here).
        use_adaptive_step_size: Whether to use adaptive step sizing (not used here).
        min_step_size: Minimum allowed step size (not used here).
        max_step_size: Maximum allowed step size (not used here).

    Returns:
        new_state: Updated state after one Euler-Maruyama step.
        new_time: time + dt.
    """
    if use_adaptive_step_size:
        raise ValueError("Adaptive step size not supported for Euler method.")

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

    return new_state, time + step_size, step_size


def sea_step(
    drift_fn: Callable,
    diffusion_fn: Callable,
    state: dict[str, ArrayLike],
    time: ArrayLike,
    step_size: ArrayLike,
    noise: dict[str, ArrayLike],
    noise_aux: dict[str, ArrayLike] = None,
    use_adaptive_step_size: bool = False,
    min_step_size: ArrayLike = None,
    max_step_size: ArrayLike = None,
) -> (dict[str, ArrayLike], ArrayLike, ArrayLike):
    """
    Performs a single shifted Euler step for SDEs with additive noise [1].

    Compared to Euler-Maruyama, this evaluates the drift at a shifted state,
    which improves the local error and the global error constant for additive noise.

    The scheme is
        X_{n+1} = X_n + f(t_n, X_n + 0.5 * g(t_n) * ΔW_n) * h + g(t_n) * ΔW_n

    [1] Foster et al., "High order splitting methods for SDEs satisfying a commutativity condition" (2023)
    Args:
        drift_fn: Function computing the drift term f(t, **state).
        diffusion_fn: Function computing the diffusion term g(t, **state).
        state: Current state, mapping variable names to tensors.
        time: Current time scalar tensor.
        step_size: Time increment dt.
        noise: Mapping of variable names to dW noise tensors.
        noise_aux: Mapping of variable names to auxiliary noise (not used here).
        use_adaptive_step_size: Whether to use adaptive step sizing (not used here).
        min_step_size: Minimum allowed step size (not used here).
        max_step_size: Maximum allowed step size (not used here).

    Returns:
        new_state: Updated state after one SEA step.
        new_time: time + dt.
    """
    if use_adaptive_step_size:
        raise ValueError("Adaptive step size not supported for Euler method.")

    # Compute diffusion (assumed additive or weakly state dependent)
    diffusion = diffusion_fn(time, **filter_kwargs(state, diffusion_fn))

    # Check noise keys
    if set(diffusion.keys()) != set(noise.keys()):
        raise ValueError("Keys of diffusion terms and noise do not match.")

    # Build shifted state: X_shift = X + 0.5 * g * ΔW
    shifted_state = {}
    for key, x in state.items():
        if key in diffusion:
            shifted_state[key] = x + 0.5 * diffusion[key] * noise[key]
        else:
            shifted_state[key] = x

    # Drift evaluated at shifted state
    drift_shifted = drift_fn(time, **filter_kwargs(shifted_state, drift_fn))

    # Final update
    new_state = {}
    for key, d in drift_shifted.items():
        base = state[key] + step_size * d
        if key in diffusion:
            base = base + diffusion[key] * noise[key]
        new_state[key] = base

    return new_state, time + step_size, step_size


def shark_step(
    drift_fn: Callable,
    diffusion_fn: Callable,
    state: Dict[str, ArrayLike],
    time: ArrayLike,
    step_size: ArrayLike,
    noise: Dict[str, ArrayLike],  # w_k = ΔW_k (already scaled by sqrt(|h|))
    noise_aux: Dict[str, ArrayLike],  # Z_k ~ N(0,1), used to build H_k
    use_adaptive_step_size: bool = False,
    min_step_size: ArrayLike = -float("inf"),
    max_step_size: ArrayLike = float("inf"),
    tolerance: float = 1e-3,
) -> Union[Tuple[Dict[str, ArrayLike], ArrayLike], Tuple[Dict[str, ArrayLike], ArrayLike, ArrayLike]]:
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
        use_adaptive_step_size: Whether to use adaptive step sizing (not used here).
        min_step_size: Minimum allowed step size (not used here).
        max_step_size: Maximum allowed step size (not used here).
        tolerance: Tolerance for adaptive step sizing.

    Returns:
        new_state: Updated state after one SHARK step.
        new_time: time + dt.
    """
    h = step_size
    t = time

    # Magnitude of the time step for stochastic scaling
    h_mag = keras.ops.abs(h)
    h_sign = keras.ops.sign(h)
    sqrt_h_mag = keras.ops.sqrt(h_mag)
    inv_sqrt3 = keras.ops.cast(1.0 / np.sqrt(3.0), dtype=keras.ops.dtype(h_mag))

    # g(y_k)
    g0 = diffusion_fn(t, **filter_kwargs(state, diffusion_fn))

    # Build H_k from w_k and Z_k
    H = {}
    for k in state.keys():
        if k in g0:
            w_k = noise[k]  # already scaled by sqrt(|h|)
            z_k = noise_aux[k]  # standard normal
            term1 = 0.5 * h_mag * w_k
            term2 = 0.5 * h_mag * sqrt_h_mag * inv_sqrt3 * z_k
            H[k] = term1 + term2
        else:
            H[k] = keras.ops.zeros_like(state[k])

    # === 1) shifted initial state ===
    y_tilde_k = {}
    for k in state.keys():
        if k in g0:
            y_tilde_k[k] = state[k] + g0[k] * H[k]
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
            sto_part = (5.0 / 6.0) * g_tilde_k[k] * noise[k]
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
            g_tilde_k[k] * ((2.0 / 5.0) * noise[k] + (6.0 / 5.0) * H[k])
            if k in g_tilde_k
            else keras.ops.zeros_like(det)
        )
        sto2 = (
            g_tilde_mid[k] * ((3.0 / 5.0) * noise[k] - (6.0 / 5.0) * H[k])
            if k in g_tilde_mid
            else keras.ops.zeros_like(det)
        )

        new_state[k] = det + sto1 + sto2

    if not use_adaptive_step_size:
        return new_state, t + h, h

    # embedded lower order solution y_low
    # here: one stage strong order one method using y_tilde_k
    y_low = {}
    for k in state.keys():
        det_low = state[k] + f_tilde_k[k] * h
        if k in g0:
            sto_low = g0[k] * noise[k]
        else:
            sto_low = keras.ops.zeros_like(det_low)
        y_low[k] = det_low + sto_low

    # error estimate as max over components of RMS norm
    err_list = []
    for k in state.keys():
        diff = new_state[k] - y_low[k]
        sq = keras.ops.square(diff)
        mean_sq = keras.ops.mean(sq)
        err_k = keras.ops.sqrt(mean_sq)
        err_list.append(err_k)

    if len(err_list) == 0:
        err = keras.ops.zeros_like(h_mag)
    else:
        err = err_list[0]
        for e_k in err_list[1:]:
            err = keras.ops.maximum(err, e_k)

    tiny = keras.ops.cast(1e12, dtype=keras.ops.dtype(h_mag))
    safety = keras.ops.cast(0.9, dtype=keras.ops.dtype(h_mag))
    # effective order between one and one point five
    exponent = keras.ops.cast(0.5, dtype=keras.ops.dtype(h_mag))

    factor = safety * keras.ops.power(tolerance / (err + tiny), exponent)

    # clamp factor
    factor_min = keras.ops.cast(0.2, dtype=keras.ops.dtype(h_mag))
    factor_max = keras.ops.cast(5.0, dtype=keras.ops.dtype(h_mag))
    factor = keras.ops.minimum(keras.ops.maximum(factor, factor_min), factor_max)

    new_h_mag = h_mag * factor
    new_h_mag = keras.ops.maximum(new_h_mag, min_step_size)
    new_h_mag = keras.ops.minimum(new_h_mag, max_step_size)

    new_h = h_sign * new_h_mag

    return new_state, t + h, new_h


def _apply_corrector(
    new_state: StateDict,
    new_time: ArrayLike,
    i: ArrayLike,
    corrector_steps: int,
    score_fn: Optional[Callable],
    step_size_factor: float,
    corrector_noise_history: Dict[str, ArrayLike],
    noise_schedule=None,
) -> StateDict:
    """Helper function to apply corrector steps."""
    if corrector_steps <= 0:
        return new_state

    # Ensures score_fn and noise_schedule are present if needed, though checked in integrate_stochastic
    if score_fn is None or noise_schedule is None:
        return new_state  # Should not happen if checks are passed

    for j in range(corrector_steps):
        score = score_fn(new_time, **filter_kwargs(new_state, score_fn))
        _z_corr = {k: corrector_noise_history[k][i, j] for k in new_state.keys()}

        log_snr_t = noise_schedule.get_log_snr(t=new_time, training=False)
        alpha_t, _ = noise_schedule.get_alpha_sigma(log_snr_t=log_snr_t)

        for k in new_state.keys():
            if k in score:
                # Calculate required norms for Langevin step
                z_norm = keras.ops.norm(_z_corr[k], axis=-1, keepdims=True)
                score_norm = keras.ops.norm(score[k], axis=-1, keepdims=True)
                score_norm = keras.ops.maximum(score_norm, 1e-8)

                # Compute step size 'e' for the Langevin update
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
    z_history: Dict[str, ArrayLike],
    z_extra_history: Dict[str, ArrayLike],
    corrector_steps: int,
    score_fn: Optional[Callable],
    step_size_factor: float,
    corrector_noise_history: Dict[str, ArrayLike],
    noise_schedule=None,
) -> StateDict:
    """
    Performs fixed-step SDE integration.
    """
    initial_step = (stop_time - start_time) / float(steps)

    def body_fixed(_i, _loop_state):
        _current_state, _current_time, _current_step = _loop_state

        # Determine step size: either the constant size or the remainder to reach stop_time
        remaining = stop_time - _current_time
        sign = keras.ops.sign(remaining)
        dt_mag = keras.ops.minimum(keras.ops.abs(_current_step), keras.ops.abs(remaining))
        dt = sign * dt_mag

        # Generate noise increment scaled by sqrt(dt)
        sqrt_dt = keras.ops.sqrt(keras.ops.abs(dt))
        _noise_i = {k: z_history[k][_i] * sqrt_dt for k in _current_state.keys()}
        if len(z_extra_history) == 0:
            _noise_extra_i = None
        else:
            _noise_extra_i = {k: z_extra_history[k][_i] for k in _current_state.keys()}

        new_state, new_time, new_step = step_fn(
            state=_current_state,
            time=_current_time,
            step_size=dt,
            noise=_noise_i,
            noise_aux=_noise_extra_i,
            use_adaptive_step_size=False,
        )

        new_state = _apply_corrector(
            new_state=new_state,
            new_time=new_time,
            i=_i,
            corrector_steps=corrector_steps,
            score_fn=score_fn,
            noise_schedule=noise_schedule,
            step_size_factor=step_size_factor,
            corrector_noise_history=corrector_noise_history,
        )
        return new_state, new_time, initial_step

    # Execute the fixed loop
    final_state, final_time, _ = keras.ops.fori_loop(0, steps, body_fixed, (state, start_time, initial_step))
    return final_state


def integrate_stochastic_adaptive(
    step_fn: Callable,
    state: StateDict,
    start_time: ArrayLike,
    stop_time: ArrayLike,
    max_steps: int,
    initial_step: ArrayLike,
    z_history: Dict[str, ArrayLike],
    z_extra_history: Dict[str, ArrayLike],
    corrector_steps: int,
    score_fn: Optional[Callable],
    step_size_factor: float,
    corrector_noise_history: Dict[str, ArrayLike],
    noise_schedule=None,
) -> StateDict:
    """
    Performs adaptive-step SDE integration.
    """
    initial_loop_state = (keras.ops.zeros((), dtype="int32"), state, start_time, initial_step, 0)

    def cond(i, current_state, current_time, current_step, counter):
        # We use a small epsilon check for floating point equality
        time_reached = keras.ops.all(keras.ops.isclose(current_time, stop_time))
        return keras.ops.logical_and(keras.ops.less(i, max_steps), keras.ops.logical_not(time_reached))

    def body_adaptive(_i, _current_state, _current_time, _current_step, _counter):
        # Step Size Control
        remaining = stop_time - _current_time
        sign = keras.ops.sign(remaining)
        # Ensure the next step does not overshoot the stop_time
        dt_mag = keras.ops.minimum(keras.ops.abs(_current_step), keras.ops.abs(remaining))
        dt = sign * dt_mag
        _counter += 1

        sqrt_dt = keras.ops.sqrt(keras.ops.abs(dt))
        _noise_i = {k: z_history[k][_i] * sqrt_dt for k in _current_state.keys()}
        if len(z_extra_history) == 0:
            _noise_extra_i = None
        else:
            _noise_extra_i = {k: z_extra_history[k][_i] for k in _current_state.keys()}

        new_state, new_time, new_step = step_fn(
            state=_current_state,
            time=_current_time,
            step_size=dt,
            noise=_noise_i,
            noise_aux=_noise_extra_i,
            use_adaptive_step_size=True,
        )

        new_state = _apply_corrector(
            new_state=new_state,
            new_time=new_time,
            i=_i,
            corrector_steps=corrector_steps,
            score_fn=score_fn,
            noise_schedule=noise_schedule,
            step_size_factor=step_size_factor,
            corrector_noise_history=corrector_noise_history,
        )

        return _i + 1, new_state, new_time, new_step, _counter

    # Execute the adaptive loop
    _, final_state, _, _, final_counter = keras.ops.while_loop(cond, body_adaptive, initial_loop_state)
    logging.debug("Finished integration after {} steps.", final_counter)
    return final_state


def integrate_stochastic(
    drift_fn: Callable,
    diffusion_fn: Callable,
    state: StateDict,
    start_time: ArrayLike,
    stop_time: ArrayLike,
    seed: keras.random.SeedGenerator,
    steps: int | Literal["adaptive"] = 100,
    method: str = "euler_maruyama",
    min_steps: int = 10,
    max_steps: int = 10_000,
    score_fn: Callable = None,
    corrector_steps: int = 0,
    noise_schedule=None,
    step_size_factor: float = 0.1,
    **kwargs,
) -> StateDict:
    """
    Integrates a stochastic differential equation from start_time to stop_time.

    Dispatches to fixed-step or adaptive-step integration logic.

    Args:
        drift_fn: Function that computes the drift term.
        diffusion_fn: Function that computes the diffusion term.
        state: Dictionary containing the initial state.
        start_time: Starting time for integration.
        stop_time: Ending time for integration. steps: Number of integration steps.
        seed: Random seed for noise generation.
        steps: Number of steps or 'adaptive' for adaptive step sizing. Only 'shark' method supports adaptive steps.
        method: Integration method to use, e.g., 'euler_maruyama' or 'shark'.
        min_steps: Minimum number of steps for adaptive integration.
        max_steps: Maximum number of steps for adaptive integration.
        score_fn: Optional score function for predictor-corrector sampling.
        corrector_steps: Number of corrector steps to take after each predictor step.
        noise_schedule: Noise schedule object for computing alpha_t in corrector.
        step_size_factor: Scaling factor for corrector step size.
        **kwargs: Additional arguments to pass to the step function.

    Returns: Final state dictionary after integration.
    """
    is_adaptive = isinstance(steps, str) and steps in ["adaptive", "dynamic"]
    if is_adaptive:
        if start_time is None or stop_time is None:
            raise ValueError("Please provide start_time and stop_time for adaptive integration.")
        if min_steps <= 0 or max_steps <= 0 or max_steps < min_steps:
            raise ValueError("min_steps and max_steps must be positive, and max_steps >= min_steps.")
        if method != "shark":
            raise ValueError("Adaptive step size is only supported for the 'shark' method.")

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

    match method:
        case "euler_maruyama":
            step_fn_raw = euler_maruyama_step
        case "sea":
            step_fn_raw = sea_step
        case "shark":
            step_fn_raw = shark_step
        case other:
            raise TypeError(f"Invalid integration method: {other!r}")

    # Partial the step function with common arguments
    step_fn = partial(
        step_fn_raw,
        drift_fn=drift_fn,
        diffusion_fn=diffusion_fn,
        min_step_size=min_step_size,
        max_step_size=max_step_size,
        **kwargs,
    )

    # Pre-generate standard normals for the predictor step (up to max_steps)
    z_history = {}
    z_extra_history = {}
    for key, val in state.items():
        shape = keras.ops.shape(val)
        z_history[key] = keras.random.normal((loop_steps, *shape), dtype=keras.ops.dtype(val), seed=seed)
        if method == "shark":
            z_extra_history[key] = keras.random.normal((loop_steps, *shape), dtype=keras.ops.dtype(val), seed=seed)

    # Pre-generate corrector noise if requested
    corrector_noise_history = {}
    if corrector_steps > 0:
        if score_fn is None or noise_schedule is None:
            raise ValueError("Please provide both score_fn and noise_schedule when using corrector_steps > 0.")
        for key, val in state.items():
            shape = keras.ops.shape(val)
            corrector_noise_history[key] = keras.random.normal(
                (loop_steps, corrector_steps, *shape), dtype=keras.ops.dtype(val), seed=seed
            )

    if is_adaptive:
        return integrate_stochastic_adaptive(
            step_fn=step_fn,
            state=state,
            start_time=start_time,
            stop_time=stop_time,
            max_steps=max_steps,
            initial_step=initial_step,
            z_history=z_history,
            z_extra_history=z_extra_history,
            corrector_steps=corrector_steps,
            score_fn=score_fn,
            noise_schedule=noise_schedule,
            step_size_factor=step_size_factor,
            corrector_noise_history=corrector_noise_history,
        )
    else:
        return integrate_stochastic_fixed(
            step_fn=step_fn,
            state=state,
            start_time=start_time,
            stop_time=stop_time,
            steps=loop_steps,
            z_history=z_history,
            z_extra_history=z_extra_history,
            corrector_steps=corrector_steps,
            score_fn=score_fn,
            noise_schedule=noise_schedule,
            step_size_factor=step_size_factor,
            corrector_noise_history=corrector_noise_history,
        )
