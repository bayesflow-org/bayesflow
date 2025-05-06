import inspect
from collections.abc import Callable, Mapping, Sequence
from functools import singledispatch
from types import FunctionType

import numpy as np

from .simulator import Simulator


@singledispatch
def make_simulator(arg, *_, **__):
    """
    This is a dispatch function that will accept a list of simulators (callables) returning
    dictionaries with simulated outputs. The outputs of simulators will be passed to following
    simulators if the latter accept keyword arguments associated with the keys of previous outputs.
    """
    raise TypeError(f"Cannot infer simulator from {arg!r}.")


@make_simulator.register
def _(simulator: Simulator):
    return simulator


@make_simulator.register(FunctionType)
def _(fn: Callable, **kwargs):
    from bayesflow.simulators import LambdaSimulator

    return LambdaSimulator(fn, **kwargs)


@make_simulator.register(Sequence)
def _(
    objs: Sequence[FunctionType],
    obj_kwargs: Mapping[str, dict[str, any]] = None,
    meta_fn: Callable[[], dict[str, np.ndarray]] = None,
    **kwargs,
):
    from bayesflow.simulators import LambdaSimulator, SequentialSimulator

    if obj_kwargs is None:
        obj_kwargs = {}

    # sanity check
    detected_names = {obj.__name__ for obj in objs if hasattr(obj, "__name__")}
    given_names = set(obj_kwargs.keys())

    if not given_names.issubset(detected_names):
        unmatched_names = given_names - detected_names
        msg = (
            f"Found at least one key in obj_kwargs that does not have a match in the object sequence:\n"
            f"{list(unmatched_names)!r}"
        )

        if not all(hasattr(obj, "__name__") for obj in objs):
            msg += (
                "\nThis can happen if the matching objects in the sequence do not have a __name__ attribute. "
                "Pass a dictionary instead to specify names explicitly."
            )

        raise ValueError(msg)

    simulators = []

    for obj in objs:
        if hasattr(obj, "__name__"):
            obj_kwargs = obj_kwargs.get(obj.__name__, {})
        else:
            obj_kwargs = {}

        simulators.append(make_simulator(obj, **obj_kwargs))

    if meta_fn is not None:
        if not inspect.signature(meta_fn).parameters:
            original_meta_fn = meta_fn

            def meta_fn(*_, **__):
                return original_meta_fn()

        meta = LambdaSimulator(meta_fn, is_batched=True)

        simulators = [meta, *simulators]

    return SequentialSimulator(simulators, **kwargs)


@make_simulator.register(Mapping)
def _(
    objs: Mapping[str, FunctionType],
    obj_kwargs: Mapping[str, dict[str, any]] = None,
    meta_fn: Callable[[], dict[str, np.ndarray]] = None,
    **kwargs,
):
    from bayesflow.simulators import LambdaSimulator, SequentialSimulator

    if obj_kwargs is None:
        obj_kwargs = {}

    # sanity check
    detected_names = set(objs.keys())
    given_names = set(obj_kwargs.keys())

    if not given_names.issubset(detected_names):
        unmatched_names = given_names - detected_names
        raise ValueError(
            f"Found at least one key in obj_kwargs that does not have a match in the object mapping:\n"
            f"{list(unmatched_names)!r}"
        )

    simulators = []

    for name, obj in objs.items():
        obj_kwargs = obj_kwargs.get(name, {})
        simulators.append(make_simulator(obj, **obj_kwargs))

    if meta_fn is not None:
        meta = LambdaSimulator(meta_fn, is_batched=True)
        simulators = [meta, *simulators]

    return SequentialSimulator(simulators, **kwargs)
