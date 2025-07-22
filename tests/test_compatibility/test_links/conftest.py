import keras
import pytest


@pytest.fixture()
def batch_size():
    return 16


@pytest.fixture()
def feature_size():
    return 10


@pytest.fixture
def link(request):
    name, kwargs = request.param
    match name:
        case "ordered":
            from bayesflow.links import Ordered

            return Ordered(**kwargs)
        case "ordered_quantiles":
            from bayesflow.links import OrderedQuantiles

            return OrderedQuantiles(**kwargs)
        case "cholesky_factor":
            from bayesflow.links import CholeskyFactor

            return CholeskyFactor(**kwargs)
        case "linear":
            return keras.layers.Activation("linear", **kwargs)
