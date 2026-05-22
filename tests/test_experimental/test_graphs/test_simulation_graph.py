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


def test_simulation_graph_output_shapes(single_level_simulator, three_level_simulator):
    shapes = single_level_simulator.graph.output_shapes()
    assert isinstance(shapes, dict)
    assert all(isinstance(v, tuple) for v in shapes.values())

    # with meta_dict to cover the substitution branch
    shapes_meta = three_level_simulator.graph.output_shapes({"N_classrooms": 5, "N_students": 10, "N_scores": 20})
    assert isinstance(shapes_meta, dict)


def test_simulation_graph_output_dimensions(single_level_simulator):
    dims = single_level_simulator.graph.output_dimensions()
    assert isinstance(dims, dict)
    assert all(isinstance(v, tuple) for v in dims.values())


def test_simulation_graph_serialization(single_level_simulator):
    graph = single_level_simulator.graph
    config = graph.get_config()
    restored = SimulationGraph.from_config(config)
    assert isinstance(restored, SimulationGraph)
    assert set(restored.nodes) == set(graph.nodes)


def test_simulation_graph_serialization_no_meta_fn():
    from bayesflow.experimental.graphical_simulator.example_simulators.single_level_simulator import prior

    simulator = GraphicalSimulator()
    simulator.add_node("prior", sample_fn=prior)
    graph = simulator.graph
    config = graph.get_config()
    restored = SimulationGraph.from_config(config)
    assert isinstance(restored, SimulationGraph)
    assert set(restored.nodes) == set(graph.nodes)
