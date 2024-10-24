import numpy as np
import pytest


def forward_transform(x):
    return x + 1


def inverse_transform(x):
    return x - 1


@pytest.fixture()
def custom_objects():
    return globals() | np.__dict__


@pytest.fixture()
def adapter():
    from bayesflow.adapters import Adapter

    d = (
        Adapter()
        .to_array()
        .convert_dtype("float64", "float32")
        .concatenate(["x1", "x2"], into="x")
        .concatenate(["y1", "y2"], into="y")
        .apply(forward=forward_transform, inverse=inverse_transform)
        # TODO: fix this in keras
        # .apply(include="p1", forward=np.log, inverse=np.exp)
        .constrain("p2", lower=0)
        .standardize()
    )

    return d


@pytest.fixture()
def random_data():
    return {
        "x1": np.random.standard_normal(size=(32, 1)),
        "x2": np.random.standard_normal(size=(32, 1)),
        "y1": np.random.standard_normal(size=(32, 2)),
        "y2": np.random.standard_normal(size=(32, 2)),
        "p1": np.random.lognormal(size=(32, 2)),
        "p2": np.random.lognormal(size=(32, 2)),
    }
