import pytest
import numpy as np

import keras

from bayesflow.utils.serialization import deserialize, serialize

import bayesflow as bf


def test_cycle_consistency(adapter, random_data):
    processed = adapter(random_data)
    deprocessed = adapter(processed, inverse=True)

    for key, value in random_data.items():
        if key in ["d1", "d2", "p3", "n1", "u1"]:
            # dropped
            continue
        assert key in deprocessed
        assert np.allclose(value, deprocessed[key])


def test_serialize_deserialize(adapter, random_data):
    processed = adapter(random_data)
    serialized = serialize(adapter)
    deserialized = deserialize(serialized)
    reserialized = serialize(deserialized)

    assert keras.tree.lists_to_tuples(serialized) == keras.tree.lists_to_tuples(reserialized)

    random_data["foo"] = random_data["x1"]
    deserialized_processed = deserialized(random_data)
    for key, value in processed.items():
        assert np.allclose(value, deserialized_processed[key])


def test_constrain():
    # check if constraint-implied transforms are applied correctly
    import numpy as np
    import warnings
    from bayesflow.adapters import Adapter

    data = {
        "x_lower_cont": np.random.exponential(1, size=(32, 1)),
        "x_upper_cont": -np.random.exponential(1, size=(32, 1)),
        "x_both_cont": np.random.beta(0.5, 0.5, size=(32, 1)),
        "x_lower_disc1": np.zeros(shape=(32, 1)),
        "x_lower_disc2": np.zeros(shape=(32, 1)),
        "x_upper_disc1": np.ones(shape=(32, 1)),
        "x_upper_disc2": np.ones(shape=(32, 1)),
        "x_both_disc1": np.vstack((np.zeros(shape=(16, 1)), np.ones(shape=(16, 1)))),
        "x_both_disc2": np.vstack((np.zeros(shape=(16, 1)), np.ones(shape=(16, 1)))),
    }

    ad = (
        Adapter()
        .constrain("x_lower_cont", lower=0)
        .constrain("x_upper_cont", upper=0)
        .constrain("x_both_cont", lower=0, upper=1)
        .constrain("x_lower_disc1", lower=0, inclusive="lower")
        .constrain("x_lower_disc2", lower=0, inclusive="none")
        .constrain("x_upper_disc1", upper=1, inclusive="upper")
        .constrain("x_upper_disc2", upper=1, inclusive="none")
        .constrain("x_both_disc1", lower=0, upper=1, inclusive="both")
        .constrain("x_both_disc2", lower=0, upper=1, inclusive="none")
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        result = ad(data)

    # continuous variables should not have boundary issues
    assert result["x_lower_cont"].min() < 0.0
    assert result["x_upper_cont"].max() > 0.0
    assert result["x_both_cont"].min() < 0.0
    assert result["x_both_cont"].max() > 1.0

    # discrete variables at the boundaries should not have issues
    # if inclusive is set properly
    assert np.isfinite(result["x_lower_disc1"].min())
    assert np.isfinite(result["x_upper_disc1"].max())
    assert np.isfinite(result["x_both_disc1"].min())
    assert np.isfinite(result["x_both_disc1"].max())

    # discrete variables at the boundaries should have issues
    # if inclusive is not set properly
    assert np.isneginf(result["x_lower_disc2"][0])
    assert np.isinf(result["x_upper_disc2"][0])
    assert np.isneginf(result["x_both_disc2"][0])
    assert np.isinf(result["x_both_disc2"][-1])


def test_simple_transforms(random_data):
    # check if simple transforms are applied correctly
    from bayesflow.adapters import Adapter

    ad = Adapter().log(["p2", "t2"]).log("t1", p1=True).sqrt("p1")

    result = ad(random_data)

    assert np.allclose(result["p2"], np.log(random_data["p2"]))
    assert np.allclose(result["t2"], np.log(random_data["t2"]))
    assert np.allclose(result["t1"], np.log1p(random_data["t1"]))
    assert np.allclose(result["p1"], np.sqrt(random_data["p1"]))

    # inverse results should match the original input
    inverse = ad(result, inverse=True)

    assert np.allclose(inverse["p2"], random_data["p2"])
    assert np.allclose(inverse["t2"], random_data["t2"])
    assert np.allclose(inverse["t1"], random_data["t1"])

    assert np.allclose(inverse["p1"], random_data["p1"])


def test_custom_transform():
    # test that transform raises errors in all relevant cases
    import keras
    from bayesflow.adapters.transforms import SerializableCustomTransform
    from copy import deepcopy

    class A:
        @classmethod
        def fn(cls, x):
            return x

    def not_registered_fn(x):
        return x

    @keras.saving.register_keras_serializable("custom")
    def registered_fn(x):
        return x

    @keras.saving.register_keras_serializable("custom")
    def registered_but_changed(x):
        return x

    def registered_but_changed(x):  # noqa: F811
        return 2 * x

    # method instead of function provided
    with pytest.raises(ValueError):
        SerializableCustomTransform(forward=A.fn, inverse=registered_fn)
    with pytest.raises(ValueError):
        SerializableCustomTransform(forward=registered_fn, inverse=A.fn)

    # lambda function provided
    with pytest.raises(ValueError):
        SerializableCustomTransform(forward=lambda x: x, inverse=registered_fn)
    with pytest.raises(ValueError):
        SerializableCustomTransform(forward=registered_fn, inverse=lambda x: x)

    # unregistered function provided
    with pytest.raises(ValueError):
        SerializableCustomTransform(forward=not_registered_fn, inverse=registered_fn)
    with pytest.raises(ValueError):
        SerializableCustomTransform(forward=registered_fn, inverse=not_registered_fn)

    # function does not match the registered function
    with pytest.raises(ValueError):
        SerializableCustomTransform(forward=registered_but_changed, inverse=registered_fn)
    with pytest.raises(ValueError):
        SerializableCustomTransform(forward=registered_fn, inverse=registered_but_changed)

    transform = SerializableCustomTransform(forward=registered_fn, inverse=registered_fn)
    serialized_transform = keras.saving.serialize_keras_object(transform)
    keras.saving.deserialize_keras_object(serialized_transform)

    # modify name of the forward function so that it cannot be found
    corrupt_serialized_transform = deepcopy(serialized_transform)
    corrupt_serialized_transform["config"]["forward"]["config"] = "nonexistent"
    with pytest.raises(TypeError):
        keras.saving.deserialize_keras_object(corrupt_serialized_transform)

    # modify the name of the inverse transform so that it cannot be found
    corrupt_serialized_transform = deepcopy(serialized_transform)
    corrupt_serialized_transform["config"]["inverse"]["config"] = "nonexistent"
    with pytest.raises(TypeError):
        keras.saving.deserialize_keras_object(corrupt_serialized_transform)


def test_split_transform(adapter, random_data):
    assert "key_to_split" in random_data

    shape = random_data["key_to_split"].shape
    target_shape = (*shape[:-1], shape[-1] // 2)

    processed = adapter(random_data)

    assert "split_1" in processed
    assert processed["split_1"].shape == target_shape

    assert "split_2" in processed
    assert processed["split_2"].shape == target_shape


def test_to_dict_transform():
    import pandas as pd

    data = {
        "int32": [1, 2, 3, 4, 5],
        "int64": [1, 2, 3, 4, 5],
        "float32": [1.0, 2.0, 3.0, 4.0, 5.0],
        "float64": [1.0, 2.0, 3.0, 4.0, 5.0],
        "object": ["a", "b", "c", "d", "e"],
        "category": ["one", "two", "three", "four", "five"],
    }

    df = pd.DataFrame(data)
    df["int32"] = df["int32"].astype("int32")
    df["int64"] = df["int64"].astype("int64")
    df["float32"] = df["float32"].astype("float32")
    df["float64"] = df["float64"].astype("float64")
    df["object"] = df["object"].astype("object")
    df["category"] = df["category"].astype("category")

    ad = bf.Adapter().to_dict()

    # drop one element to simulate non-complete data
    batch = df.iloc[:-1]

    processed = ad(batch)

    assert isinstance(processed, dict)
    assert list(processed.keys()) == ["int32", "int64", "float32", "float64", "object", "category"]

    for key, value in processed.items():
        assert isinstance(value, np.ndarray)
        assert value.dtype == "float32"

    # category should have 5 one-hot categories, even though it was only passed 4
    assert processed["category"].shape[-1] == 5


def test_log_det_jac(adapter_log_det_jac, random_data):
    d, log_det_jac = adapter_log_det_jac(random_data, log_det_jac=True)

    assert np.allclose(log_det_jac["x1"], np.log(2))

    p1 = -np.log1p(random_data["p1"])
    p2 = -0.5 * np.log(random_data["p2"]) - np.log(2)
    p3 = random_data["p3"] - np.log(np.exp(random_data["p3"]) - 1)
    p = np.sum(p1, axis=-1) + np.sum(p2, axis=-1) + np.sum(p3, axis=-1)

    assert np.allclose(log_det_jac["p"], p)

    n1 = -(random_data["n1"] - 1)
    n1 = n1 - np.log(np.exp(n1) - 1)
    n1 = np.sum(n1, axis=-1)

    assert np.allclose(log_det_jac["n1"], n1)

    u1 = random_data["u1"]
    u1 = (u1 + 1) / 3
    u1 = -np.log(u1) - np.log1p(-u1) - np.log(3)

    assert np.allclose(log_det_jac["u"], u1[:, 0])


def test_log_det_jac_inverse(adapter_log_det_jac_inverse, random_data):
    d, forward_log_det_jac = adapter_log_det_jac_inverse(random_data, log_det_jac=True)
    d, inverse_log_det_jac = adapter_log_det_jac_inverse(d, inverse=True, log_det_jac=True)

    for key in forward_log_det_jac.keys():
        assert np.allclose(forward_log_det_jac[key], -inverse_log_det_jac[key])


def test_log_det_jac_exceptions(random_data):
    # Test cannot compute inverse log_det_jac
    # e.g., when we apply a concat and then a transform that
    # we cannot "unconcatenate" the log_det_jac
    # (because the log_det_jac are summed, not concatenated)
    adapter = bf.Adapter().concatenate(["p1", "p2", "p3"], into="p").sqrt("p")
    transformed_data, log_det_jac = adapter(random_data, log_det_jac=True)

    # test that inverse raises error
    with pytest.raises(ValueError):
        adapter(transformed_data, inverse=True, log_det_jac=True)

    # test resolvable order: first transform, then concatenate
    adapter = bf.Adapter().sqrt(["p1", "p2", "p3"]).concatenate(["p1", "p2", "p3"], into="p")

    transformed_data, forward_log_det_jac = adapter(random_data, log_det_jac=True)
    data, inverse_log_det_jac = adapter(transformed_data, inverse=True, log_det_jac=True)
    inverse_log_det_jac = sum(inverse_log_det_jac.values())

    # forward is the same regardless
    assert np.allclose(forward_log_det_jac["p"], log_det_jac["p"])

    # inverse works when concatenation is used after transforms
    assert np.allclose(forward_log_det_jac["p"], -inverse_log_det_jac)
