import pytest

from bayesflow.experimental.graphical_simulator import GraphicalSimulator
from bayesflow.experimental.graphs import SimulationGraph, ExpandedGraph, InvertedGraph


@pytest.mark.parametrize(
    "simulator",
    ["single_level_simulator", "two_level_simulator", "three_level_simulator", "crossed_design_irt_simulator"],
)
def test_simulation_graph(request, simulator):
    simulator = request.getfixturevalue(simulator)
    graph = simulator.graph

    assert isinstance(simulator, GraphicalSimulator)
    assert isinstance(graph, SimulationGraph)

    assert isinstance(graph.expand(), ExpandedGraph)
    assert isinstance(graph.invert(), InvertedGraph)
    assert isinstance(graph.variable_names(), dict)
    assert isinstance(graph.data_node(), str)
