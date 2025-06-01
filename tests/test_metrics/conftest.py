import pytest


@pytest.fixture()
def root_mean_squared_error():
    from bayesflow.metrics import RootMeanSquaredError

    return RootMeanSquaredError(normalize=True, name="rmse", dtype="float64")


@pytest.fixture()
def maximum_mean_discrepancy():
    from bayesflow.metrics import MaximumMeanDiscrepancy

    return MaximumMeanDiscrepancy(name="mmd", kernel="gaussian", unbiased=True, dtype="float64")


@pytest.fixture(params=["root_mean_squared_error", "maximum_mean_discrepancy"])
def metric(request):
    return request.getfixturevalue(request.param)
