import pytest
import numpy as np


@pytest.fixture()
def metric(request):
    name, kwargs = request.param

    match name:
        case "root_mean_squared_error":
            from bayesflow.metrics import RootMeanSquaredError

            return RootMeanSquaredError(**kwargs)
        case "maximum_mean_discrepancy":
            from bayesflow.metrics import MaximumMeanDiscrepancy

            return MaximumMeanDiscrepancy(**kwargs)
    raise ValueError(f"unknown name: {name}")


@pytest.fixture
def samples_1():
    rng = np.random.default_rng(seed=1)
    return rng.normal(size=(2, 3)).astype(np.float32)


@pytest.fixture
def samples_2():
    rng = np.random.default_rng(seed=2)
    return rng.normal(size=(2, 3)).astype(np.float32)
