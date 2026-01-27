from bayesflow.experimental.graphical_simulator.example_simulators import two_level_simulator
from bayesflow.adapters import Adapter
from bayesflow.networks import DeepSet, CouplingFlow
from bayesflow.experimental.graphical_approximator import GraphicalApproximator


def two_level_approximator():
    simulator = two_level_simulator()

    adapter = Adapter()
    adapter.to_array()
    adapter.convert_dtype("float64", "float32")

    summary_networks = [DeepSet(summary_dim=10), DeepSet(summary_dim=20)]
    inference_networks = [CouplingFlow(), CouplingFlow()]

    inverted_graph = simulator.graph.invert()
    approximator = GraphicalApproximator(
        inverted_graph, adapter=adapter, inference_networks=inference_networks, summary_networks=summary_networks
    )

    return approximator
