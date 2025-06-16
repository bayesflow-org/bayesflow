import pytest
import numpy as np


@pytest.fixture()
def default_adapter():
    from bayesflow import Adapter

    return Adapter.create_default(["x1", "x2"])


@pytest.fixture()
def complete_adapter():
    from bayesflow.adapters import Adapter
    import keras

    @keras.saving.register_keras_serializable("custom")
    def serializable_fn(x):
        return x

    return (
        Adapter()
        .to_array()
        .as_set(["s1", "s2"])
        .broadcast("t1", to="t2")
        .as_time_series(["t1", "t2"])
        .convert_dtype("float64", "float32", exclude="o1")
        .concatenate(["x1", "x2"], into="x")
        .concatenate(["y1", "y2"], into="y")
        .expand_dims(["z1"], axis=2)
        .squeeze("z1", axis=2)
        .log("p1")
        .constrain("p2", lower=0)
        .apply(include="p2", forward="exp", inverse="log")
        .apply(include="p2", forward="log1p")
        .apply_serializable(include="x", forward=serializable_fn, inverse=serializable_fn)
        .scale("x", by=[-1, 2])
        .shift("x", by=2)
        .split("key_to_split", into=["split_1", "split_2"])
        .standardize(exclude=["t1", "t2", "o1"])
        .drop("d1")
        .one_hot("o1", 10)
        .keep(["x", "y", "z1", "p1", "p2", "s1", "s2", "s3", "t1", "t2", "o1", "split_1", "split_2"])
        .rename("o1", "o2")
        .random_subsample("s3", sample_size=33, axis=0)
        .take("s3", indices=np.arange(0, 32), axis=0)
        .group(["p1", "p2"], into="ps", prefix="p")
        .ungroup("ps", prefix="p")
    )


@pytest.fixture(params=["default_adapter", "complete_adapter"])
def adapter(request):
    return request.getfixturevalue(request.param)


def get_data(rng):
    return {
        "x1": rng.standard_normal(size=(32, 1)),
        "x2": rng.standard_normal(size=(32, 1)),
        "y1": rng.standard_normal(size=(32, 2)),
        "y2": rng.standard_normal(size=(32, 2)),
        "z1": rng.standard_normal(size=(32, 2)),
        "p1": rng.lognormal(size=(32, 2)),
        "p2": rng.lognormal(size=(32, 2)),
        "p3": rng.lognormal(size=(32, 2)),
        "n1": 1 - rng.lognormal(size=(32, 2)),
        "s1": rng.standard_normal(size=(32, 3, 2)),
        "s2": rng.standard_normal(size=(32, 3, 2)),
        "t1": np.zeros((3, 2)),
        "t2": np.ones((32, 3, 2)),
        "d1": rng.standard_normal(size=(32, 2)),
        "d2": rng.standard_normal(size=(32, 2)),
        "o1": rng.integers(0, 9, size=(32, 2)),
        "s3": rng.standard_normal(size=(35, 2)),
        "u1": rng.uniform(low=-1, high=2, size=(32, 1)),
        "key_to_split": rng.standard_normal(size=(32, 10)),
    }


@pytest.fixture
def data_1():
    rng = np.random.default_rng(seed=1)
    return get_data(rng)


@pytest.fixture
def data_2():
    rng = np.random.default_rng(seed=2)
    return get_data(rng)
