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
        if key == "s3":
            # we subsampled this key, so it is expected for its shape to change
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
        if key == "s3":
            # skip this key because it is *randomly* subsampled
            continue

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


def test_nan_to_num():
    arr = {"test": np.array([1.0, np.nan, 3.0])}
    # test without mask
    transform = bf.Adapter().nan_to_num(keys="test", default_value=-1.0, return_mask=False)
    out = transform.forward(arr)["test"]
    np.testing.assert_array_equal(out, np.array([1.0, -1.0, 3.0]))

    # test with mask
    arr = {"test": np.array([1.0, np.nan, 3.0]), "test-2d": np.array([[1.0, np.nan], [np.nan, 4.0]])}
    transform = bf.Adapter().nan_to_num(keys="test", default_value=0.0, return_mask=True)
    out = transform.forward(arr)
    np.testing.assert_array_equal(out["test"], np.array([1.0, 0.0, 3.0]))
    np.testing.assert_array_equal(out["mask_test"], np.array([1.0, 0.0, 1.0]))

    # test two-d array
    transform = bf.Adapter().nan_to_num(keys="test-2d", default_value=0.5, return_mask=True, mask_prefix="new_mask")
    out = transform.forward(arr)
    np.testing.assert_array_equal(out["test-2d"], np.array([[1.0, 0.5], [0.5, 4.0]]))
    np.testing.assert_array_equal(out["new_mask_test-2d"], np.array([[1, 0], [0, 1]]))


def test_nnpe(random_data):
    # NNPE cannot be integrated into the adapter fixture and its tests since it modifies the input data
    # and therefore breaks existing allclose checks
    import numpy as np
    from bayesflow.adapters import Adapter

    # Test basic case with global noise application
    ad = Adapter().nnpe("x1", spike_scale=1.0, slab_scale=1.0, per_dimension=False, seed=42)
    result_training = ad(random_data, stage="training")
    result_validation = ad(random_data, stage="validation")
    result_inference = ad(random_data, stage="inference")
    result_inversed = ad(random_data, inverse=True)
    serialized = serialize(ad)
    deserialized = deserialize(serialized)
    reserialized = serialize(deserialized)

    assert keras.tree.lists_to_tuples(serialized) == keras.tree.lists_to_tuples(reserialized)

    # check that only x1 is changed
    assert "x1" in result_training
    assert not np.allclose(result_training["x1"], random_data["x1"])

    # all other keys are untouched
    for k, v in random_data.items():
        if k == "x1":
            continue
        assert np.allclose(result_training[k], v)

    # check that the validation and inference data as well as inversed results are unchanged
    for k, v in random_data.items():
        assert np.allclose(result_validation[k], v)
        assert np.allclose(result_inference[k], v)
        assert np.allclose(result_inversed[k], v)

    # Test both scales and seed are None case (automatic scale determination) with dimensionwise noise application
    ad_auto = Adapter().nnpe("y1", slab_scale=None, spike_scale=None, per_dimension=True, seed=None)
    result_training_auto = ad_auto(random_data, stage="training")
    assert not np.allclose(result_training_auto["y1"], random_data["y1"])
    for k, v in random_data.items():
        if k == "y1":
            continue
        assert np.allclose(result_training_auto[k], v)

    serialized_auto = serialize(ad_auto)
    deserialized_auto = deserialize(serialized_auto)
    reserialized_auto = serialize(deserialized_auto)
    assert keras.tree.lists_to_tuples(serialized_auto) == keras.tree.lists_to_tuples(serialize(reserialized_auto))

    # Test dimensionwise versus global noise application (per_dimension=True vs per_dimension=False)
    # Create data with second dimension having higher variance
    data_shape = (32, 16, 1)
    rng = np.random.default_rng(42)
    zero = np.ones(shape=data_shape)
    high = rng.normal(0, 100.0, size=data_shape)
    var_data = {"x": np.concatenate([zero, high], axis=-1)}

    # Apply dimensionwise and global adapters with automatic slab_scale scale determination
    ad_partial_global = Adapter().nnpe("x", spike_scale=0, slab_scale=None, per_dimension=False, seed=42)
    ad_partial_dim = Adapter().nnpe("x", spike_scale=[0, 1], slab_scale=None, per_dimension=True, seed=42)
    res_dim = ad_partial_dim(var_data, stage="training")
    res_glob = ad_partial_global(var_data, stage="training")

    # Compute standard deviations of noise per last axis dimension
    noise_dim = res_dim["x"] - var_data["x"]
    noise_glob = res_glob["x"] - var_data["x"]
    std_dim = np.std(noise_dim, axis=(0, 1))
    std_glob = np.std(noise_glob, axis=(0, 1))

    # Dimensionwise should assign zero noise, global some noise to zero-variance dimension
    assert std_dim[0] == 0
    assert std_glob[0] > 0
    # Both should assign noise to high-variance dimension
    assert std_dim[1] > 0
    assert std_glob[1] > 0
