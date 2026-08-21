from collections.abc import Mapping
import functools
import inspect
import random as random_module

import numpy as np


def next_seed_sequence(seed_sequence: np.random.SeedSequence) -> np.random.SeedSequence:
    return seed_sequence.spawn(1)[0]


def next_uint32(seed_sequence: np.random.SeedSequence) -> int:
    return int(next_seed_sequence(seed_sequence).generate_state(1, dtype=np.uint32)[0])


def reseed_generator(rng: np.random.Generator, seed_sequence: np.random.SeedSequence) -> None:
    bit_generator_type = type(rng.bit_generator)
    rng.bit_generator.state = bit_generator_type(next_seed_sequence(seed_sequence)).state


def reseed_random_state(root: object, seed_sequence: np.random.SeedSequence) -> None:
    """Reseed RNGs reachable from a worker-local object copy."""

    random_module.seed(next_uint32(seed_sequence))
    np.random.seed(next_uint32(seed_sequence))

    seen = set()

    def visit(obj):
        obj_id = id(obj)
        if obj_id in seen:
            return

        seen.add(obj_id)

        if isinstance(obj, np.random.Generator):
            reseed_generator(obj, seed_sequence)
            return

        if isinstance(obj, np.random.RandomState):
            obj.seed(next_uint32(seed_sequence))
            return

        if isinstance(obj, random_module.Random):
            obj.seed(next_uint32(seed_sequence))
            return

        if obj is None or isinstance(obj, (str, bytes, int, float, complex, bool, np.ndarray)):
            return

        if inspect.ismodule(obj) or inspect.isclass(obj):
            return

        if inspect.ismethod(obj):
            visit(obj.__self__)
            visit(obj.__func__)
            return

        if inspect.isfunction(obj):
            visit(obj.__defaults__)
            visit(obj.__kwdefaults__)
            visit(getattr(obj, "__dict__", None))

            if obj.__closure__ is not None:
                for cell in obj.__closure__:
                    try:
                        visit(cell.cell_contents)
                    except ValueError:
                        pass

            for name in obj.__code__.co_names:
                if name in obj.__globals__:
                    visit(obj.__globals__[name])

            return

        if isinstance(obj, functools.partial):
            visit(obj.func)
            visit(obj.args)
            visit(obj.keywords)
            return

        if isinstance(obj, Mapping):
            for value in obj.values():
                visit(value)
            return

        if isinstance(obj, (list, tuple, set, frozenset)):
            for value in obj:
                visit(value)
            return

        try:
            visit(vars(obj))
        except TypeError:
            pass

        slots = getattr(type(obj), "__slots__", ())
        if isinstance(slots, str):
            slots = (slots,)

        for slot in slots:
            try:
                visit(getattr(obj, slot))
            except AttributeError:
                pass

    visit(root)
