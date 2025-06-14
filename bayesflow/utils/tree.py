import optree
from typing import Callable


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


def map_dict(func: Callable, dictionary: dict) -> dict:
    """Applies a function to all leaves of a (possibly nested) dictionary.

    Parameters
    ----------
    func : Callable
        The function to apply to the leaves.
    dictionary : dict
        The input dictionary.

    Returns
    -------
    dict
        A dictionary with the outputs of `func` as leaves.
    """

    def is_not_dict(x):
        return not isinstance(x, dict)

    return optree.tree_map(
        func,
        dictionary,
        is_leaf=is_not_dict,
        none_is_leaf=True,
        namespace="keras",
    )
