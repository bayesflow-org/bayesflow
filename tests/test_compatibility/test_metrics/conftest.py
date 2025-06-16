import pytest
import numpy as np


@pytest.fixture()
def root_mean_squared_error():
    from bayesflow.metrics import RootMeanSquaredError

    return RootMeanSquaredError(normalize=True, name="rmse", dtype="float32")


@pytest.fixture()
def maximum_mean_discrepancy():
    from bayesflow.metrics import MaximumMeanDiscrepancy

    return MaximumMeanDiscrepancy(name="mmd", kernel="gaussian", unbiased=True, dtype="float32")


@pytest.fixture(params=["root_mean_squared_error", "maximum_mean_discrepancy"])
def metric(request):
    return request.getfixturevalue(request.param)


@pytest.fixture
def samples_1():
    rng = np.random.default_rng(seed=1)
    return rng.normal(size=(2, 3))


@pytest.fixture
def samples_2():
    rng = np.random.default_rng(seed=2)
    return rng.normal(size=(2, 3))
