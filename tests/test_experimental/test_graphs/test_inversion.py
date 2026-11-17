import numpy as np
import pytest

from bayesflow.experimental.graphs import InvertedGraph, SimulationGraph
from bayesflow.experimental.graphs.factorizations import mean_repetitions


def group_reps(size):
    return np.random.randint(2, 5, size=size)


def test_networks_merge_roots(two_level_simulator):
    expanded = two_level_simulator.graph.expand()

    assert expanded.invert(["hypers", "shared", "locals"]).networks() == [["hypers", "shared"], ["locals"]]


def test_conditions(two_level_simulator):
    inverted = two_level_simulator.graph.expand().invert(["hypers", "shared", "locals"])

    assert inverted.conditions(["hypers", "shared"]) == ["y"]
    assert inverted.conditions(["locals"]) == ["hypers", "shared", "y"]

    # shared is inferred after locals, so locals cannot condition on it
    inverted = two_level_simulator.graph.expand().invert(["hypers", "locals", "shared"])
    assert inverted.conditions(["locals"]) == ["hypers", "y"]


def test_locals_condition_on_their_own_data_only(two_level_simulator):
    inverted = two_level_simulator.graph.expand().invert(["hypers", "shared", "locals"])

    for node in inverted:
        if node.original == "locals":
            data = [parent for parent in inverted.predecessors(node) if parent.original == "y"]
            assert [parent.index for parent in data] == [node.index]


def test_outer_nodes_first_is_amortizable(two_level_simulator):
    inverted = two_level_simulator.graph.expand().invert(["hypers", "shared", "locals"])

    assert inverted.is_amortizable("locals")


def test_inner_nodes_first_is_not_amortizable(two_level_simulator):
    inverted = two_level_simulator.graph.expand().invert(["locals", "hypers", "shared"])

    assert not inverted.is_amortizable("locals")


def test_default_ordering_infers_outer_nodes_first(two_level_simulator, three_level_simulator):
    assert two_level_simulator.graph.expand().invert().networks() == [["hypers", "shared"], ["locals"]]

    networks = three_level_simulator.graph.expand().invert().networks()
    assert networks == [["schools", "shared"], ["classrooms"], ["students"]]


def test_crossed_design_infers_smaller_factor_sequentially(crossed_design_irt_simulator):
    # 15 questions vs. 100-200 students per school
    inverted = crossed_design_irt_simulator.graph.expand().invert()

    assert inverted.is_amortizable("students")
    assert not inverted.is_amortizable("questions")


def test_mean_repetitions():
    graph = SimulationGraph(meta_fn=lambda: {"num_groups": 4})
    graph.add_node("fixed", reps=3)
    graph.add_node("from_meta", reps="num_groups")
    graph.add_node("from_callable", reps=lambda size: np.full(size, 7))
    graph.add_node("default")
    graph.add_node("data")
    for node in ["fixed", "from_meta", "from_callable", "default"]:
        graph.add_edge(node, "data")

    assert mean_repetitions(graph) == {"fixed": 3, "from_meta": 4, "from_callable": 7, "default": 1}


def test_config_round_trip():
    graph = SimulationGraph()
    graph.add_node("hypers", reps=1)
    graph.add_node("groups", reps=group_reps)
    graph.add_node("y", reps="num_obs")
    graph.add_edge("hypers", "groups")
    graph.add_edge("groups", "y")
    inverted = graph.expand().invert(["hypers", "groups"])

    restored = InvertedGraph.from_config(inverted.get_config())

    assert restored.ordering == inverted.ordering
    assert set(restored.edges) == set(inverted.edges)
    assert restored.expanded_graph.simulation_graph.nodes["groups"]["reps"] is group_reps


def test_config_rejects_local_reps_functions():
    graph = SimulationGraph()
    graph.add_node("hypers", reps=1)
    graph.add_node("y", reps=lambda size: np.full(size, 3))
    graph.add_edge("hypers", "y")

    with pytest.raises(ValueError, match="top-level"):
        graph.expand().invert(["hypers"]).get_config()
