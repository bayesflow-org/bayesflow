import pytest
import sympy as sp


def test_concatenate_shapes():
    from bayesflow.experimental.graphical_approximator.shape_operations import concatenate_shapes

    assert concatenate_shapes([(2, 3), (2, 3), (2, 10, 5)]) == (2, 10, 11)
    a = (sp.Symbol("B"), 1)
    b = (sp.Symbol("B"), sp.Symbol("N"), 3)
    c = (sp.Symbol("B"), sp.Symbol("N"), 2)

    assert concatenate_shapes([a, b, c]) == (
        sp.Symbol("B"),
        sp.Symbol("N"),
        6,
    )


def test_resolve_shapes():
    from bayesflow.experimental.graphical_approximator.shape_operations import resolve_shapes

    a = (sp.Symbol("B"), sp.Symbol("N_groups"), 3)
    b = (sp.Symbol("B"), 3)

    assert resolve_shapes({"beta": a, "sigma": b}, {"B": 2, "N_groups": 3}) == {
        "beta": (2, 3, 3),
        "sigma": (2, 3),
    }
    assert resolve_shapes({"beta": a, "sigma": b}, {"N_groups": 3}) == {
        "beta": (sp.Symbol("B"), 3, 3),
        "sigma": (sp.Symbol("B"), 3),
    }


def test_replace_placeholders():
    from bayesflow.experimental.graphical_approximator.shape_operations import replace_placeholders

    a = (sp.Symbol("B"), sp.Symbol("N"), 3)
    assert replace_placeholders(a, {"B": 4, "N": 3}) == (4, 3, 3)
    assert replace_placeholders((1, 2, 3), {"B": 4, "N": 3}) == (1, 2, 3)


def test_expand_shape_rank():
    from bayesflow.experimental.graphical_approximator.shape_operations import expand_shape_rank

    assert expand_shape_rank((2, 1, 20), 5) == (2, 1, 1, 1, 20)
    assert expand_shape_rank((sp.Symbol("B"), 2), 4) == (sp.Symbol("B"), 1, 1, 2)
    with pytest.raises(ValueError):
        expand_shape_rank((1, 2, 3, 4), 3)


def test_stack_shapes():
    from bayesflow.experimental.graphical_approximator.shape_operations import stack_shapes

    assert stack_shapes((2, 20), (2, 3, 3)) == (2, 3, 23)
    assert stack_shapes((sp.Symbol("B"), 3), (sp.Symbol("B"), 1)) == (sp.Symbol("B"), 4)
    with pytest.raises(ValueError):
        stack_shapes((sp.Symbol("B"), 3), (2, 3))
