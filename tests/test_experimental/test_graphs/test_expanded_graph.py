import pytest

from bayesflow.experimental.graphical_simulator import GraphicalSimulator
from bayesflow.experimental.graphs import ExpandedGraph, InvertedGraph


@pytest.mark.parametrize(
    "simulator",
    ["single_level_simulator", "two_level_simulator", "three_level_simulator", "crossed_design_irt_simulator"],
)
def test_expanded_graph(request, simulator):
    simulator = request.getfixturevalue(simulator)
    graph = simulator.graph
    expanded_graph = graph.expand()

    assert isinstance(simulator, GraphicalSimulator)
    assert isinstance(expanded_graph, ExpandedGraph)

    assert isinstance(expanded_graph.invert(), InvertedGraph)
