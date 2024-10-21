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

    def contexts():
        r = np.random.normal(0.1, 0.01)
        alpha = np.random.uniform(-0.5 * np.pi, 0.5 * np.pi)
        return dict(r=r, alpha=alpha)

    def parameters():
        return dict(theta=np.random.uniform(-1.0, 1.0, size=2))

    def observables(r, alpha, theta):
        x1 = -np.abs(theta[0] + theta[1]) / np.sqrt(2.0) + r * np.cos(alpha) + 0.25
        x2 = (-theta[0] + theta[1]) / np.sqrt(2.0) + r * np.sin(alpha)
        return dict(x=np.stack([x1, x2]))

    return make_simulator([contexts, parameters, observables])


@pytest.fixture(params=["composite_two_moons", "two_moons"])
def simulator(request):
    return request.getfixturevalue(request.param)


@pytest.fixture()
def two_moons():
    from bayesflow.simulators import TwoMoons

    return TwoMoons()
