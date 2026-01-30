import pytest
from bayesflow.networks import DeepSet, CouplingFlow
from bayesflow.adapters import Adapter
from bayesflow.experimental.graphical_approximator import GraphicalApproximator


@pytest.fixture()
def single_level_simulator():
    from bayesflow.experimental.graphical_simulator.example_simulators import single_level_simulator

    return single_level_simulator()


@pytest.fixture()
def single_level_approximator():
    from bayesflow.experimental.graphical_simulator.example_simulators import single_level_simulator

    simulator = single_level_simulator()

    adapter = Adapter()
    adapter.to_array()
    adapter.convert_dtype("float64", "float32")

    summary_networks = [DeepSet(summary_dim=10)]
    inference_networks = [CouplingFlow()]

    inverted_graph = simulator.graph.invert()
    approximator = GraphicalApproximator(
        inverted_graph, adapter=adapter, inference_networks=inference_networks, summary_networks=summary_networks
    )

    return approximator


@pytest.fixture()
def two_level_simulator():
    from bayesflow.experimental.graphical_simulator.example_simulators import two_level_simulator

    return two_level_simulator()


@pytest.fixture()
def two_level_approximator():
    from bayesflow.experimental.graphical_simulator.example_simulators import two_level_simulator

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


@pytest.fixture()
def two_level_repeated_roots_simulator():
    from bayesflow.experimental.graphical_simulator.example_simulators import two_level_simulator

    return two_level_simulator(repeated_roots=True)


@pytest.fixture()
def two_level_repeated_roots_approximator():
    from bayesflow.experimental.graphical_simulator.example_simulators import two_level_simulator

    simulator = two_level_simulator(repeated_roots=True)

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


@pytest.fixture()
def three_level_simulator():
    from bayesflow.experimental.graphical_simulator.example_simulators import three_level_simulator

    return three_level_simulator()


@pytest.fixture()
def three_level_approximator():
    from bayesflow.experimental.graphical_simulator.example_simulators import three_level_simulator

    simulator = three_level_simulator()

    adapter = Adapter()
    adapter.to_array()
    adapter.convert_dtype("float64", "float32")

    summary_networks = [DeepSet(summary_dim=10), DeepSet(summary_dim=20), DeepSet(summary_dim=30)]
    inference_networks = [CouplingFlow(), CouplingFlow(), CouplingFlow()]

    inverted_graph = simulator.graph.invert()
    approximator = GraphicalApproximator(
        inverted_graph, adapter=adapter, inference_networks=inference_networks, summary_networks=summary_networks
    )

    return approximator


@pytest.fixture()
def crossed_design_irt_simulator():
    from bayesflow.experimental.graphical_simulator.example_simulators import crossed_design_irt_simulator

    return crossed_design_irt_simulator()


@pytest.fixture()
def crossed_design_irt_approximator():
    from bayesflow.experimental.graphical_simulator.example_simulators import crossed_design_irt_simulator

    simulator = crossed_design_irt_simulator()

    adapter = Adapter()
    adapter.to_array()
    adapter.convert_dtype("float64", "float32")

    summary_networks = [DeepSet(summary_dim=10), DeepSet(summary_dim=20)]
    inference_networks = [CouplingFlow(), CouplingFlow(), CouplingFlow()]

    inverted_graph = simulator.graph.invert()
    approximator = GraphicalApproximator(
        inverted_graph, adapter=adapter, inference_networks=inference_networks, summary_networks=summary_networks
    )

    return approximator
