from bayesflow.experimental.graphs.utils import (
    extract_subgraph,
    has_open_path,
    merge_nodes,
    merge_root_nodes,
    sort_nodes_topologically,
)
from copy import deepcopy
import networkx as nx


def test_extract_subgraph(three_level_graph):
    #    schools
    #       |
    #       |
    #  classrooms
    #       |
    #       |     shared
    #   students    /
    #        \     /
    #         \   /
    #        scores
    g = three_level_graph.simulation_graph
    g2 = deepcopy(g)

    subgraph = extract_subgraph(g, "schools")
    expected = nx.DiGraph()
    expected.add_edges_from([("schools", "classrooms"), ("classrooms", "students"), ("students", "scores")])
    assert nx.is_isomorphic(subgraph, expected)

    subgraph = extract_subgraph(g, node="classrooms")
    expected = nx.DiGraph()
    expected.add_edges_from([("classrooms", "students"), ("students", "scores")])
    assert nx.is_isomorphic(subgraph, expected)

    subgraph = extract_subgraph(g, node="students")
    expected = nx.DiGraph()
    expected.add_edges_from([("students", "scores")])
    assert nx.is_isomorphic(subgraph, expected)

    subgraph = extract_subgraph(g, node="scores")
    expected = nx.DiGraph()
    expected.add_node("scores")
    assert nx.is_isomorphic(subgraph, expected)

    subgraph = extract_subgraph(g, node="shared")
    expected = nx.DiGraph()
    expected.add_edges_from([("shared", "scores")])
    assert nx.is_isomorphic(subgraph, expected)

    assert nx.is_isomorphic(g, g2)  # extract_subgraph does not mutate


def test_has_open_path():
    # A -> B -> C
    g = nx.DiGraph()
    g.add_edges_from([("A", "B"), ("B", "C")])

    # path A--B--C is open if we don't condition on B
    assert has_open_path(g, "A", "C", known=[]) is True
    assert has_open_path(g, "A", "C", known=["B"]) is False

    # A <- B -> C
    g = nx.DiGraph()
    g.add_edges_from([("B", "A"), ("B", "C")])

    # path is open unless conditioned on B
    assert has_open_path(g, "A", "C", known=[]) is True
    assert has_open_path(g, "A", "C", known=["B"]) is False

    # A -> B <- C
    g = nx.DiGraph()
    g.add_edges_from([("A", "B"), ("C", "B")])

    # collider is blocked unless conditioned on B or descendant of B
    assert has_open_path(g, "A", "C", known=[]) is False
    assert has_open_path(g, "A", "C", known=["B"]) is True

    g.add_edges_from([("B", "D")])
    assert has_open_path(g, "A", "C", known=[]) is False
    assert has_open_path(g, "A", "C", known=["D"]) is True

    # stays blocked when conditioned on unrelated node
    g.add_edges_from([("C", "E")])
    assert has_open_path(g, "A", "C", known=["E"]) is False

    # Two undirected-simple paths between A and C:
    # Path 1: A -> B -> C (chain) (blocked if B known)
    # Path 2: A -> D <- C (collider) (blocked unless D or descendant known)
    #
    # If we condition on B only, Path 1 blocks but Path 2 is still blocked (collider not conditioned) => False
    # If we condition on B and D, Path 1 blocks but Path 2 opens => True
    g = nx.DiGraph()
    g.add_edges_from(
        [
            ("A", "B"),
            ("B", "C"),  # chain
            ("A", "D"),
            ("C", "D"),  # collider at D
        ]
    )

    assert has_open_path(g, "A", "C", known=["B"]) is False
    assert has_open_path(g, "A", "C", known=["B", "D"]) is True


def test_sort_nodes_topologically(two_level_graph):
    g = two_level_graph.simulation_graph
    g2 = deepcopy(g)

    nodes = ["y", "locals", "shared", "hypers"]

    assert sort_nodes_topologically(g, nodes) == ["hypers", "locals", "shared", "y"]
    assert nx.is_isomorphic(g, g2)  # sort_nodes_topologically does not mutate


def test_merge_root_nodes(two_level_graph):
    #  hypers
    #     |
    #  locals  shared
    #      \    /
    #       \  /
    #        y
    g = two_level_graph.simulation_graph
    g2 = deepcopy(g)

    merged = merge_root_nodes(g)

    expected = nx.DiGraph()
    expected.add_edges_from([("locals", "y"), ("hypers, shared", "locals"), ("hypers, shared", "y")])

    print(merged.edges)
    print(expected.edges)

    assert nx.is_isomorphic(merged, expected)
    assert nx.is_isomorphic(g, g2)  # merge_root_nodes does not mutate


def test_merge_nodes():
    #    A
    #   / \
    #  B   D
    #   \ /
    #    C
    g = nx.DiGraph()
    g.add_edges_from([("A", "B"), ("B", "C"), ("A", "D"), ("D", "C")])
    g2 = deepcopy(g)

    merged = merge_nodes(g, ["B", "D"])

    expected = nx.DiGraph()
    expected.add_edges_from([("A", "B, D"), ("B, D", "C")])

    assert nx.is_isomorphic(merged, expected)
    assert nx.is_isomorphic(g, g2)  # merge_nodes does not mutate
