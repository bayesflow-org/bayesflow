import numpy as np
import pytest


def forward_transform(x):
    return x + 1


def inverse_transform(x):
    return x - 1


@pytest.fixture()
def custom_objects():
    return dict(forward_transform=forward_transform, inverse_transform=inverse_transform)


@pytest.fixture()
def adapter():
    from bayesflow.adapters import Adapter

    adapter = (
        Adapter()
        .to_array()
        .as_set(["s1", "s2"])
        .broadcast("t1", to="t2")
        .as_time_series(["t1", "t2"])
        .convert_dtype("float64", "float32", exclude="o1")
        .concatenate(["x1", "x2"], into="x")
        .concatenate(["y1", "y2"], into="y")
        .expand_dims(["z1"], axis=2)
        .apply(forward=forward_transform, inverse=inverse_transform)
        # TODO: fix this in keras
        # .apply(include="p1", forward=np.log, inverse=np.exp)
        .constrain("p2", lower=0)
        .standardize(exclude=["t1", "t2", "o1"])
        .drop("d1")
        .one_hot("o1", 10)
        .keep(["x", "y", "z1", "p1", "p2", "s1", "s2", "t1", "t2", "o1"])
        .rename("o1", "o2")
    )

    return adapter


@pytest.fixture()
def random_data():
    return {
        "x1": np.random.standard_normal(size=(32, 1)),
        "x2": np.random.standard_normal(size=(32, 1)),
        "y1": np.random.standard_normal(size=(32, 2)),
        "y2": np.random.standard_normal(size=(32, 2)),
        "z1": np.random.standard_normal(size=(32, 2)),
        "p1": np.random.lognormal(size=(32, 2)),
        "p2": np.random.lognormal(size=(32, 2)),
        "s1": np.random.standard_normal(size=(32, 3, 2)),
        "s2": np.random.standard_normal(size=(32, 3, 2)),
        "t1": np.zeros((3, 2)),
        "t2": np.ones((32, 3, 2)),
        "d1": np.random.standard_normal(size=(32, 2)),
        "d2": np.random.standard_normal(size=(32, 2)),
        "o1": np.random.randint(0, 9, size=(32, 2)),
    }
