import numpy as np
import pytest


@pytest.fixture()
def batch_size():
    return 16


@pytest.fixture(params=[False, True], autouse=True)
def use_batched(request):
    return request.param


@pytest.fixture(params=[False, True], autouse=True)
def use_numpy(request):
    return request.param


@pytest.fixture(params=[False, True], autouse=True)
def use_squeezed(request):
    return request.param


@pytest.fixture()
def composite_two_moons():
    from bayesflow.simulators import make_simulator

    def parameters():
        parameters = np.random.uniform(-1.0, 1.0, size=2)
        return dict(parameters=parameters)

    def observables(parameters):
        r = np.random.normal(0.1, 0.01)
        alpha = np.random.uniform(-0.5 * np.pi, 0.5 * np.pi)
        x1 = -np.abs(parameters[0] + parameters[1]) / np.sqrt(2.0) + r * np.cos(alpha) + 0.25
        x2 = (-parameters[0] + parameters[1]) / np.sqrt(2.0) + r * np.sin(alpha)
        return dict(observables=np.stack([x1, x2]))

    return make_simulator([parameters, observables])


@pytest.fixture(params=["composite_two_moons", "two_moons"])
def simulator(request):
    return request.getfixturevalue(request.param)


@pytest.fixture()
def two_moons():
    from bayesflow.simulators import TwoMoons

    return TwoMoons()
