from copy import deepcopy

import networkx as nx

from bayesflow.experimental.graphs.utils import (
    add_previous_names_metadata,
    add_split_by_metadata,
    add_suffix,
    extract_subgraph,
    has_open_path,
    merge_nodes,
    merge_root_nodes,
    sort_nodes_topologically,
    split_node,
)


def test_split_node(three_level_graph):
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

    # splitting on classrooms means that from schools, there are now two paths:
    # schools -> classrooms_1 -> students_1 -> scores_1 and
    # schools -> classrooms_2 -> students_2 -> scores_2
    split_graph = split_node(g, "classrooms")

    expected = nx.DiGraph()
    expected.add_edges_from(
        [
            ("schools", "classrooms_1"),
            ("classrooms_1", "students_1"),
            ("students_1", "scores_1"),
            ("schools", "classrooms_2"),
            ("classrooms_2", "students_2"),
            ("students_2", "scores_2"),
            ("shared", "scores_1"),
            ("shared", "scores_2"),
        ]
    )

    assert nx.is_isomorphic(split_graph, expected)
    assert nx.is_isomorphic(g, g2)  # split_graph is not mutating

    expected_previous_names = {
        "classrooms_1": ["classrooms"],
        "classrooms_2": ["classrooms"],
        "students_1": ["students"],
        "students_2": ["students"],
        "scores_1": ["scores"],
        "scores_2": ["scores"],
    }

    print(split_graph.nodes["scores_1"])
    for node, previous_names in expected_previous_names.items():
        assert split_graph.nodes[node]["previous_names"] == previous_names
        assert split_graph.nodes[node]["split_by"] == ["classrooms"]

    # second split:
    # classrooms_1 -> students_11 -> scores_11 and
    # classrooms_1 -> students_12 -> scores_12
    split_graph_2 = split_node(split_graph, "classrooms_1")

    expected = nx.DiGraph()
    expected.add_edges_from(
        [
            ("schools", "classrooms_11"),
            ("schools", "classrooms_12"),
            ("classrooms_11", "students_11"),
            ("students_11", "scores_11"),
            ("classrooms_12", "students_12"),
            ("students_12", "scores_12"),
            ("schools", "classrooms_2"),
            ("classrooms_2", "students_2"),
            ("students_2", "scores_2"),
            ("shared", "scores_11"),
            ("shared", "scores_12"),
            ("shared", "scores_2"),
        ]
    )

    assert nx.is_isomorphic(split_graph_2, expected)
    expected_previous_names = {
        "classrooms_11": ["classrooms", "classrooms_1"],
        "classrooms_12": ["classrooms", "classrooms_1"],
        "classrooms_2": ["classrooms"],
        "students_11": ["students", "students_1"],
        "students_12": ["students", "students_1"],
        "students_2": ["students"],
        "scores_11": ["scores", "scores_1"],
        "scores_12": ["scores", "scores_1"],
        "scores_2": ["scores"],
    }

    for node, previous_names in expected_previous_names.items():
        assert split_graph_2.nodes[node]["previous_names"] == previous_names

    expected_split_by = {
        "classrooms_11": ["classrooms", "classrooms_1"],
        "classrooms_12": ["classrooms", "classrooms_1"],
        "classrooms_2": ["classrooms"],
        "students_11": ["classrooms", "classrooms_1"],
        "students_12": ["classrooms", "classrooms_1"],
        "students_2": ["classrooms"],
        "scores_11": ["classrooms", "classrooms_1"],
        "scores_12": ["classrooms", "classrooms_1"],
        "scores_2": ["classrooms"],
    }

    for node, split_by in expected_split_by.items():
        assert split_graph_2.nodes[node]["split_by"] == split_by


def test_add_previous_names_metadata():
    # A1 -> B1 -> C1
    g1 = nx.DiGraph()
    g1.add_edges_from([("A1", "B1"), ("B1", "C1")])
    g1.nodes["A1"]["previous_names"] = ["A"]

    # A2 -> B2 -> C2
    g2 = nx.DiGraph()
    g2.add_edges_from([("A2", "B2"), ("B2", "C2")])

    with_metadata = add_previous_names_metadata(g1, g2, "A1", "A2")
    assert with_metadata.nodes["A2"]["previous_names"] == ["A", "A1"]

    with_metadata = add_previous_names_metadata(g1, g2, "B1", "B2")
    assert with_metadata.nodes["B2"]["previous_names"] == ["B1"]


def test_add_split_by_metadata(three_level_graph):
    # A1 -> B1 -> C1
    g1 = nx.DiGraph()
    g1.add_edges_from([("A1", "B1"), ("B1", "C1")])
    g1.nodes["A1"]["split_by"] = ["A"]

    # A2 -> B2 -> C2
    g2 = nx.DiGraph()
    g2.add_edges_from([("A2", "B2"), ("B2", "C2")])

    with_metadata = add_split_by_metadata(g1, g2, "A1", "A2", "S1")
    assert with_metadata.nodes["A2"]["split_by"] == ["A", "S1"]

    with_metadata = add_split_by_metadata(g1, g2, "B1", "B2", "S1")
    assert with_metadata.nodes["B2"]["split_by"] == ["S1"]


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

    assert nx.is_isomorphic(g, g2)  # extract_subgraph is not mutating


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


def test_add_suffix():
    assert add_suffix("node_1", 2) == "node_12"
    assert add_suffix("node", suffix=3) == "node_3"


def test_sort_nodes_topologically(two_level_graph):
    g = two_level_graph.simulation_graph
    g2 = deepcopy(g)

    nodes = ["y", "locals", "shared", "hypers"]

    assert sort_nodes_topologically(g, nodes) == ["hypers", "locals", "shared", "y"]
    assert nx.is_isomorphic(g, g2)  # sort_nodes_topologically is not mutating


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
    assert nx.is_isomorphic(g, g2)  # merge_root_nodes is not mutating


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
    assert nx.is_isomorphic(g, g2)  # merge_nodes is not mutating
