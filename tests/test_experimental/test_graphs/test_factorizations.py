import itertools
import math

import pytest

from bayesflow.experimental.graphs import InvertedGraph
from bayesflow.experimental.graphs.factorizations import (
    enumerate_factorizations,
    node_name_mapping,
    select_factorization,
    to_expanded_nodes,
)

SIMULATORS = [
    "single_level_simulator",
    "two_level_simulator",
    "three_level_simulator",
    "crossed_design_irt_simulator",
]


@pytest.fixture()
def three_level_expanded_graph(three_level_simulator):
    return three_level_simulator.graph.expand()


def latent_groups(expanded_graph):
    """The node groups that are inferred, i.e. everything that is not a leaf."""
    mapping = node_name_mapping(expanded_graph)

    return [group for group, nodes in mapping.items() if any(expanded_graph.out_degree(n) != 0 for n in nodes)]


def inferred_nodes(factorization):
    return set(itertools.chain.from_iterable(factorization.network_composition().values()))


def num_sequentially_inferred(factorization):
    return len(inferred_nodes(factorization) - set(factorization.amortizable_nodes()))


def test_to_expanded_nodes():
    node_mapping = {("a",): ["a_1", "a_2"], ("b", "c"): ["b, c"], ("d",): ["d_1"]}

    assert to_expanded_nodes([("a",), ("d",)], node_mapping) == ["a_1", "a_2", "d_1"]
    assert to_expanded_nodes([("d",), ("a",)], node_mapping) == ["d_1", "a_1", "a_2"]
    assert to_expanded_nodes([("b", "c")], node_mapping) == ["b, c"]
    assert to_expanded_nodes([], node_mapping) == []


def test_to_expanded_nodes_unknown_group():
    with pytest.raises(KeyError):
        to_expanded_nodes([("missing",)], {("a",): ["a_1"]})


def test_node_name_mapping_two_level(two_level_simulator):
    expanded_graph = two_level_simulator.graph.expand()

    # hypers and shared are both roots and get merged, locals and y are split in two
    assert node_name_mapping(expanded_graph) == {
        ("hypers", "shared"): ["hypers, shared"],
        ("locals",): ["locals_1", "locals_2"],
        ("y",): ["y_1", "y_2"],
    }


def test_node_name_mapping_three_level(three_level_expanded_graph):
    assert node_name_mapping(three_level_expanded_graph) == {
        ("schools", "shared"): ["schools, shared"],
        ("classrooms",): ["classrooms_1", "classrooms_2"],
        ("students",): ["students_1", "students_2"],
        ("scores",): ["scores_1", "scores_2"],
    }


@pytest.mark.parametrize("simulator", SIMULATORS)
def test_node_name_mapping_groups_every_expanded_node_once(request, simulator):
    expanded_graph = request.getfixturevalue(simulator).graph.expand()

    mapping = node_name_mapping(expanded_graph)

    # keys are used as dictionary keys downstream, so they must be tuples of names
    for group in mapping:
        assert isinstance(group, tuple)
        assert all(isinstance(name, str) for name in group)

    # the groups partition the expanded graph: every node belongs to exactly one
    grouped_nodes = list(itertools.chain.from_iterable(mapping.values()))
    assert sorted(grouped_nodes) == sorted(expanded_graph.nodes)


@pytest.mark.parametrize(
    "simulator, expected",
    [
        ("single_level_simulator", 1),
        ("two_level_simulator", 2),
        ("three_level_simulator", 6),
        ("crossed_design_irt_simulator", 6),
    ],
)
def test_enumerate_factorizations_count(request, simulator, expected):
    expanded_graph = request.getfixturevalue(simulator).graph.expand()

    factorizations = enumerate_factorizations(expanded_graph)

    assert len(factorizations) == expected
    assert all(isinstance(factorization, InvertedGraph) for factorization in factorizations)


@pytest.mark.parametrize("simulator", SIMULATORS)
def test_enumerate_factorizations_covers_every_ordering(request, simulator):
    expanded_graph = request.getfixturevalue(simulator).graph.expand()

    factorizations = enumerate_factorizations(expanded_graph)

    # one factorization per ordering of the latent groups
    assert len(factorizations) == math.factorial(len(latent_groups(expanded_graph)))


def test_enumerate_factorizations_is_deterministic(three_level_expanded_graph):
    first = enumerate_factorizations(three_level_expanded_graph)
    second = enumerate_factorizations(three_level_expanded_graph)

    assert [f.network_composition() for f in first] == [f.network_composition() for f in second]


def test_select_factorization_single_candidate(single_level_simulator):
    factorizations = enumerate_factorizations(single_level_simulator.graph.expand())

    assert len(factorizations) == 1
    assert select_factorization(factorizations) is factorizations[0]


def test_select_factorization_two_level(two_level_simulator):
    selected = select_factorization(enumerate_factorizations(two_level_simulator.graph.expand()))

    assert selected.network_composition() == {0: ["hypers", "shared"], 1: ["locals"]}
    assert len(selected.summary_network_input_shapes()) == 3


def test_select_factorization_three_level(three_level_expanded_graph):
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
    selected = select_factorization(enumerate_factorizations(three_level_expanded_graph))

    assert selected.network_composition() == {0: ["schools", "shared"], 1: ["students"], 2: ["classrooms"]}
    assert len(selected.summary_network_input_shapes()) == 5


def test_select_factorization_crossed_design_irt(crossed_design_irt_simulator):
    factorizations = enumerate_factorizations(crossed_design_irt_simulator.graph.expand())

    selected = select_factorization(factorizations)

    # questions and students are crossed, so one of them is always inferred sequentially
    assert selected.network_composition() == {0: ["schools"], 1: ["questions"], 2: ["students"]}
    assert num_sequentially_inferred(selected) == 1
