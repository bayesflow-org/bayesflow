import pytest


@pytest.fixture()
def single_level_simulator():
    from bayesflow.experimental.graphical_simulator.example_simulators import single_level_simulator

    return single_level_simulator()


@pytest.fixture()
def two_level_simulator():
    from bayesflow.experimental.graphical_simulator.example_simulators import two_level_simulator

    return two_level_simulator()


@pytest.fixture()
def two_level_repeated_roots_simulator():
    from bayesflow.experimental.graphical_simulator.example_simulators import two_level_simulator

    return two_level_simulator(repeated_roots=True)


@pytest.fixture()
def three_level_simulator():
    from bayesflow.experimental.graphical_simulator.example_simulators import three_level_simulator

    return three_level_simulator()


@pytest.fixture()
def crossed_design_irt_simulator():
    from bayesflow.experimental.graphical_simulator.example_simulators import crossed_design_irt_simulator

    return crossed_design_irt_simulator()
