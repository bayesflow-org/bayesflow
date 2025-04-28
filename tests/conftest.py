import keras
import logging
import matplotlib
import pytest

BACKENDS = ["jax", "numpy", "tensorflow", "torch"]


def pytest_runtest_setup(item):
    """Skips backends by test markers. Unmarked tests are treated as backend-agnostic"""
    backend = keras.backend.backend()

    test_backends = [mark.name for mark in item.iter_markers() if mark.name in BACKENDS]

    if test_backends and backend not in test_backends:
        pytest.skip(f"Skipping backend '{backend}' for test {item}, which is registered for backends {test_backends}.")

    # always show full tracebacks
    keras.config.disable_traceback_filtering()

    if keras.backend.backend() == "jax":
        import jax

        jax.config.update("jax_traceback_filtering", "off")

    # use a non-GUI plotting backend for tests
    matplotlib.use("Agg")

    # set the logging level to debug for all tests
    logging.getLogger("bayesflow").setLevel(logging.DEBUG)


def pytest_runtest_teardown(item, nextitem):
    import matplotlib.pyplot as plt

    # close all plots at the end of each test
    plt.close("all")


def pytest_make_parametrize_id(config, val, argname):
    return f"{argname}={repr(val)}"


@pytest.fixture(params=[2], scope="session")
def batch_size(request):
    return request.param


@pytest.fixture(params=[None, 2, 3], scope="session")
def conditions_size(request):
    return request.param


@pytest.fixture(params=[1, 4], scope="session")
def summary_dim(request):
    return request.param


@pytest.fixture(params=["two_moons"], scope="session")
def dataset(request):
    return request.getfixturevalue(request.param)


@pytest.fixture(params=[2, 3], scope="session")
def feature_size(request):
    return request.param


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
