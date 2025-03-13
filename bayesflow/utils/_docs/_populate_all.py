import inspect


def _add_imports_to_all(include_modules: bool | list[str] = False, exclude: list[str] | None = None):
    """Add all global variables to __all__"""
    assert type(include_modules) in [bool, list]
    exclude = exclude or []
    calling_module = inspect.stack()[1]
    local_stack = calling_module[0]
    global_vars = local_stack.f_globals
    all_vars = global_vars["__all__"] if "__all__" in global_vars else []
    included_vars = []
    for var_name in set(global_vars.keys()):
        if inspect.ismodule(global_vars[var_name]):
            if include_modules is True and var_name not in exclude and not var_name.startswith("_"):
                included_vars.append(var_name)
            elif isinstance(include_modules, list) and var_name in include_modules:
                included_vars.append(var_name)
        elif var_name not in exclude and not var_name.startswith("_"):
            included_vars.append(var_name)
    global_vars["__all__"] = sorted(list(set(all_vars).union(included_vars)))
