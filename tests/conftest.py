import keras
import pytest


BACKENDS = ["jax", "numpy", "tensorflow", "torch"]


def pytest_runtest_setup(item):
    """Skips backends by test markers. Unmarked tests are treated as backend-agnostic"""
    backend = keras.backend.backend()

    test_backends = [mark.name for mark in item.iter_markers() if mark.name in BACKENDS]

    if test_backends and backend not in test_backends:
        pytest.skip(f"Skipping backend '{backend}' for test {item}, which is registered for backends {test_backends}.")


def pytest_make_parametrize_id(config, val, argname):
    return f"{argname}={repr(val)}"


@pytest.fixture(params=[2, 3], scope="session", autouse=True)
def batch_size(request):
    return request.param


@pytest.fixture(params=[None, 2, 3], scope="session")
def conditions_size(request):
    return request.param


@pytest.fixture(scope="function")
def coupling_flow():
    from bayesflow.networks import CouplingFlow

    return CouplingFlow(depth=2, subnet="mlp", subnet_kwargs=dict(widths=(32, 32)))


@pytest.fixture(params=["two_moons"], scope="session")
def dataset(request):
    return request.getfixturevalue(request.param)


@pytest.fixture(params=[2, 3], scope="session")
def feature_size(request):
    return request.param


@pytest.fixture(params=["coupling_flow"], scope="function")
def inference_network(request):
    return request.getfixturevalue(request.param)


@pytest.fixture(params=["inference_network", "summary_network"], scope="function")
def network(request):
    return request.getfixturevalue(request.param)


@pytest.fixture(scope="session")
def random_conditions(batch_size, conditions_size):
    if conditions_size is None:
        return None

    return keras.random.normal((batch_size, conditions_size))


@pytest.fixture(scope="session")
def random_samples(batch_size, feature_size):
    return keras.random.normal((batch_size, feature_size))


@pytest.fixture(scope="function", autouse=True)
def random_seed():
    seed = 0
    keras.utils.set_random_seed(seed)
    return seed


@pytest.fixture(scope="session")
def random_set(batch_size, set_size, feature_size):
    return keras.random.normal((batch_size, set_size, feature_size))


@pytest.fixture(params=[2, 3], scope="session")
def set_size(request):
    return request.param


@pytest.fixture(params=["two_moons"], scope="session")
def simulator(request):
    return request.getfixturevalue(request.param)


@pytest.fixture(params=[None], scope="function")
def summary_network(request):
    if request.param is None:
        return None
    return request.getfixturevalue(request.param)


@pytest.fixture(scope="session")
def training_dataset(simulator, batch_size):
    from bayesflow.datasets import OfflineDataset

    num_batches = 128
    samples = simulator.sample((num_batches * batch_size,))
    return OfflineDataset(samples, batch_size=batch_size)


@pytest.fixture(scope="session")
def two_moons(batch_size):
    from bayesflow.simulators import TwoMoonsSimulator

    return TwoMoonsSimulator()


@pytest.fixture(scope="session")
def validation_dataset(simulator, batch_size):
    from bayesflow.datasets import OfflineDataset

    num_batches = 16
    samples = simulator.sample((num_batches * batch_size,))
    return OfflineDataset(samples, batch_size=batch_size)
