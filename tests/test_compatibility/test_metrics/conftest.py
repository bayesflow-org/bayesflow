import pytest
import keras


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
    return keras.random.normal((2, 3), seed=1)


@pytest.fixture
def samples_2():
    return keras.random.normal((2, 3), seed=2)
