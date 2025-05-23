import numpy as np
import pytest


@pytest.fixture()
def adapter():
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
        "p3": np.random.lognormal(size=(32, 2)),
        "n1": 1 - np.random.lognormal(size=(32, 2)),
        "s1": np.random.standard_normal(size=(32, 3, 2)),
        "s2": np.random.standard_normal(size=(32, 3, 2)),
        "t1": np.zeros((3, 2)),
        "t2": np.ones((32, 3, 2)),
        "d1": np.random.standard_normal(size=(32, 2)),
        "d2": np.random.standard_normal(size=(32, 2)),
        "o1": np.random.randint(0, 9, size=(32, 2)),
        "s3": np.random.standard_normal(size=(35, 2)),
        "u1": np.random.uniform(low=-1, high=2, size=(32, 1)),
        "key_to_split": np.random.standard_normal(size=(32, 10)),
    }


@pytest.fixture()
def adapter_log_det_jac():
    from bayesflow.adapters import Adapter

    return (
        Adapter()
        .scale("x1", by=2)
        .log("p1", p1=True)
        .sqrt("p2")
        .constrain("p3", lower=0)
        .constrain("n1", upper=1)
        .constrain("u1", lower=-1, upper=2)
        .concatenate(["p1", "p2", "p3"], into="p")
        .rename("u1", "u")
    )


@pytest.fixture()
def adapter_log_det_jac_inverse():
    from bayesflow.adapters import Adapter

    return (
        Adapter()
        .standardize("x1", mean=1, std=2)
        .log("p1")
        .sqrt("p2")
        .constrain("p3", lower=0, method="log")
        .constrain("n1", upper=1, method="log")
        .constrain("u1", lower=-1, upper=2)
        .scale(["p1", "p2", "p3"], by=3.5)
    )
