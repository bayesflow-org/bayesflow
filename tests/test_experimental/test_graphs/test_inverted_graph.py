import pytest


def test_network_conditions(single_level_graph, two_level_graph, three_level_graph, crossed_design_irt_graph):
    assert single_level_graph.network_conditions() == {0: ["likelihood"]}

    assert two_level_graph.network_conditions() == {0: ["y"], 1: ["hypers", "shared", "y"]}

    assert three_level_graph.network_conditions() == {
        0: ["scores"],
        1: ["schools", "shared", "scores"],
        2: ["schools", "classrooms", "shared", "scores"],
    }

    assert crossed_design_irt_graph.network_conditions() == {
        0: ["observations"],
        1: ["schools", "observations"],
        2: ["schools", "questions", "observations"],
    }


def test_network_compositions(single_level_graph, two_level_graph, three_level_graph, crossed_design_irt_graph):
    assert single_level_graph.network_composition() == {0: ["prior"]}

    assert two_level_graph.network_composition() == {0: ["hypers", "shared"], 1: ["locals"]}

    assert three_level_graph.network_composition() == {
        0: ["schools", "shared"],
        1: ["classrooms"],
        2: ["students"],
    }

    assert crossed_design_irt_graph.network_composition() == {
        0: ["schools"],
        1: ["questions"],
        2: ["students"],
    }


def test_data_shape_order(single_level_graph, two_level_graph, three_level_graph, crossed_design_irt_graph):
    assert single_level_graph.data_shape_order() == []
    assert two_level_graph.data_shape_order() == ["locals", "y"]
    assert three_level_graph.data_shape_order() == ["classrooms", "students", "scores"]
    assert crossed_design_irt_graph.data_shape_order() == ["questions", "students"]


def test_permutated_data_shape_order(single_level_graph, two_level_graph, three_level_graph, crossed_design_irt_graph):
    assert single_level_graph.permutated_data_shape_order() == []
    assert two_level_graph.permutated_data_shape_order() == ["locals", "y"]
    assert three_level_graph.permutated_data_shape_order() == ["classrooms", "students", "scores"]
    assert crossed_design_irt_graph.permutated_data_shape_order() == ["students", "questions"]


def test_amortizable_nodes(single_level_graph, two_level_graph, three_level_graph, crossed_design_irt_graph):
    assert single_level_graph.amortizable_nodes() == ["prior"]
    assert two_level_graph.amortizable_nodes() == ["hypers", "locals", "shared"]
    assert three_level_graph.amortizable_nodes() == ["schools", "classrooms", "students", "shared"]
    assert crossed_design_irt_graph.amortizable_nodes() == ["schools", "students"]


def test_allows_amortization(single_level_graph, two_level_graph, three_level_graph, crossed_design_irt_graph):
    for graph in [single_level_graph, two_level_graph, three_level_graph, crossed_design_irt_graph]:
        for node in graph.amortizable_nodes():
            assert graph.allows_amortization(node)

    with pytest.raises(ValueError):
        single_level_graph.allows_amortization("non existing node name")


def test_original_node_names(single_level_graph, two_level_graph, three_level_graph, crossed_design_irt_graph):
    assert single_level_graph.original_node_names() == {"likelihood": "likelihood", "prior": "prior"}
    assert two_level_graph.original_node_names() == {
        "y_1": "y",
        "y_2": "y",
        "hypers, shared": ["hypers", "shared"],
        "locals_1": "locals",
        "locals_2": "locals",
    }
    assert three_level_graph.original_node_names() == {
        "scores_1": "scores",
        "scores_2": "scores",
        "schools, shared": ["schools", "shared"],
        "classrooms_1": "classrooms",
        "classrooms_2": "classrooms",
        "students_1": "students",
        "students_2": "students",
    }
    assert crossed_design_irt_graph.original_node_names() == {
        "observations_11": "observations",
        "observations_21": "observations",
        "observations_12": "observations",
        "observations_22": "observations",
        "schools": "schools",
        "questions_1": "questions",
        "questions_2": "questions",
        "students_1": "students",
        "students_2": "students",
    }


def test_conditions_by_node(single_level_graph, two_level_graph, three_level_graph, crossed_design_irt_graph):
    assert single_level_graph.conditions_by_node() == {"likelihood": [], "prior": ["likelihood"]}
    assert two_level_graph.conditions_by_node() == {
        "y": [],
        "hypers": ["y"],
        "shared": ["y"],
        "locals": ["hypers", "shared", "y"],
    }
    assert three_level_graph.conditions_by_node() == {
        "scores": [],
        "schools": ["scores"],
        "shared": ["scores"],
        "classrooms": ["schools", "shared", "scores"],
        "students": ["schools", "classrooms", "shared", "scores"],
    }
    assert crossed_design_irt_graph.conditions_by_node() == {
        "observations": [],
        "schools": ["observations"],
        "questions": ["schools", "questions", "observations"],
        "students": ["schools", "questions", "observations"],
    }


def test_detailed_conditions_by_node(single_level_graph, two_level_graph, three_level_graph, crossed_design_irt_graph):
    assert single_level_graph.detailed_conditions_by_node() == {"likelihood": [], "prior": ["likelihood"]}
    assert two_level_graph.detailed_conditions_by_node() == {
        "y_1": [],
        "y_2": [],
        "hypers, shared": ["y_1", "y_2"],
        "locals_1": ["hypers, shared", "y_1"],
        "locals_2": ["hypers, shared", "y_2"],
    }
    assert three_level_graph.detailed_conditions_by_node() == {
        "scores_1": [],
        "scores_2": [],
        "schools, shared": ["scores_1", "scores_2"],
        "classrooms_1": ["schools, shared", "scores_1"],
        "classrooms_2": ["schools, shared", "scores_2"],
        "students_1": ["classrooms_1", "scores_1", "schools, shared"],
        "students_2": ["classrooms_2", "scores_2", "schools, shared"],
    }
    assert crossed_design_irt_graph.detailed_conditions_by_node() == {
        "observations_11": [],
        "observations_21": [],
        "observations_12": [],
        "observations_22": [],
        "schools": ["observations_11", "observations_12", "observations_21", "observations_22"],
        "questions_1": ["observations_11", "observations_12", "schools", "observations_21", "observations_22"],
        "questions_2": [
            "observations_21",
            "observations_22",
            "schools",
            "questions_1",
            "observations_11",
            "observations_12",
        ],
        "students_1": ["observations_11", "observations_21", "schools", "questions_1", "questions_2"],
        "students_2": ["observations_12", "observations_22", "schools", "questions_1", "questions_2"],
    }
