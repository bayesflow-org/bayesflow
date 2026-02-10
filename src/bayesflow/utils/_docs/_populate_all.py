import sys
import types


def _add_imports_to_all(include_modules: bool | list[str] = False, exclude: list[str] | None = None):
    """Add all global variables to __all__"""
    if not isinstance(include_modules, (bool, list)):
        raise ValueError("include_modules must be a boolean or a list of strings")

    exclude_set = set(exclude or [])
    contains = exclude_set.__contains__
    mod_type = types.ModuleType
    frame = sys._getframe(1)
    g: dict = frame.f_globals
    existing = set(g.get("__all__", []))

    to_add = []
    include_list = include_modules if isinstance(include_modules, list) else ()
    inc_all = include_modules is True

    for name, val in g.items():
        if name.startswith("_") or contains(name):
            continue

        if isinstance(val, mod_type):
            if inc_all or name in include_list:
                to_add.append(name)
        else:
            to_add.append(name)

    g["__all__"] = sorted(existing.union(to_add))
