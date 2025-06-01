import optree


def flatten_shape(structure):
    def is_shape_tuple(x):
        return isinstance(x, (list, tuple)) and all(isinstance(e, (int, type(None))) for e in x)

    leaves, _ = optree.tree_flatten(
        structure,
        is_leaf=is_shape_tuple,
        none_is_leaf=True,
        namespace="keras",
    )
    return leaves


def map_dict(func, *structures):
    def is_not_dict(x):
        return not isinstance(x, dict)

    if not structures:
        raise ValueError("Must provide at least one structure")

    # Add check for same structures, otherwise optree just maps to shallowest.
    def func_with_check(*args):
        if not all(optree.tree_is_leaf(s, is_leaf=is_not_dict, none_is_leaf=True, namespace="keras") for s in args):
            raise ValueError("Structures don't have the same nested structure.")
        return func(*args)

    map_func = func_with_check if len(structures) > 1 else func

    return optree.tree_map(
        map_func,
        *structures,
        is_leaf=is_not_dict,
        none_is_leaf=True,
        namespace="keras",
    )


def map_dict_with_path(func, *structures):
    def is_not_dict(x):
        return not isinstance(x, dict)

    if not structures:
        raise ValueError("Must provide at least one structure")

    # Add check for same structures, otherwise optree just maps to shallowest.
    def func_with_check(*args):
        if not all(optree.tree_is_leaf(s, is_leaf=is_not_dict, none_is_leaf=True, namespace="keras") for s in args):
            raise ValueError("Structures don't have the same nested structure.")
        return func(*args)

    map_func = func_with_check if len(structures) > 1 else func

    return optree.tree_map_with_path(
        map_func,
        *structures,
        is_leaf=is_not_dict,
        none_is_leaf=True,
        namespace="keras",
    )


def get_value_at_path(structure, path):
    output = structure
    for accessor in path:
        output = output.__getitem__(accessor)
    return output
