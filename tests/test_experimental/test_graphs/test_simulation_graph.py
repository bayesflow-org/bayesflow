import pytest

from bayesflow.experimental.graphs import SimulationGraph
from bayesflow.experimental.graphs.simulation_graph import is_exchangeable

SIMULATORS = ["single_level_simulator", "two_level_simulator", "three_level_simulator", "crossed_design_irt_simulator"]


def test_expand_num_nodes(
    single_level_simulator, two_level_simulator, three_level_simulator, crossed_design_irt_simulator
):
    assert single_level_simulator.graph.expand().number_of_nodes() == 2
    assert two_level_simulator.graph.expand().number_of_nodes() == 6
    assert three_level_simulator.graph.expand().number_of_nodes() == 12
    assert crossed_design_irt_simulator.graph.expand().number_of_nodes() == 9


@pytest.mark.parametrize("simulator", SIMULATORS)
def test_expand_preserves_dependencies(request, simulator):
    graph = request.getfixturevalue(simulator).graph
    expanded = graph.expand()

    assert {(u.original, w.original) for u, w in expanded.edges} == set(graph.edges)


def test_expand_splits_into_two_instances(two_level_simulator):
    expanded = two_level_simulator.graph.expand()

    locals_ = [node for node in expanded if node.original == "locals"]
    assert sorted(node.index for node in locals_) == [(("locals", 1),), (("locals", 2),)]

    # each copy of y belongs to exactly one copy of locals
    for node in locals_:
        children = list(expanded.successors(node))
        assert [child.original for child in children] == ["y"]
        assert children[0].index == node.index


def test_expand_skips_single_repetition(two_level_simulator):
    two_level_simulator.graph.nodes["locals"]["reps"] = 1

    assert two_level_simulator.graph.expand().number_of_nodes() == 4


def test_is_exchangeable():
    graph = SimulationGraph()
    graph.add_node("single", reps=1)
    graph.add_node("fixed", reps=5)
    graph.add_node("from_meta", reps="num_groups")
    graph.add_node("from_callable", reps=lambda: 3)

    # every node that can have more than one instance is exchangeable
    assert not is_exchangeable(graph, "single")
    assert is_exchangeable(graph, "fixed")
    assert is_exchangeable(graph, "from_meta")
    assert is_exchangeable(graph, "from_callable")
