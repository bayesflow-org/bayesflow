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
