from keras.saving import (
    deserialize_keras_object as deserialize,
    serialize_keras_object as serialize,
)
import numpy as np


def test_cycle_consistency(adapter, random_data):
    processed = adapter(random_data)
    deprocessed = adapter(processed, inverse=True)

    for key, value in random_data.items():
        if key in ["d1", "d2"]:
            # dropped
            continue
        assert key in deprocessed
        assert np.allclose(value, deprocessed[key])


def test_serialize_deserialize(adapter, random_data):
    processed = adapter(random_data)
    serialized = serialize(adapter)
    deserialized = deserialize(serialized)
    reserialized = serialize(deserialized)

    assert reserialized.keys() == serialized.keys()
    for key in reserialized:
        assert reserialized[key] == serialized[key]

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

    assert np.array_equal(result["p2"], np.log(random_data["p2"]))
    assert np.array_equal(result["t2"], np.log(random_data["t2"]))
    assert np.array_equal(result["t1"], np.log1p(random_data["t1"]))
    assert np.array_equal(result["p1"], np.sqrt(random_data["p1"]))

    # inverse results should match the original input
    inverse = ad(result, inverse=True)

    assert np.array_equal(inverse["p2"], random_data["p2"])
    assert np.array_equal(inverse["t2"], random_data["t2"])
    assert np.array_equal(inverse["t1"], random_data["t1"])
    # numerical inaccuries prevent np.array_equal to work here
    assert np.allclose(inverse["p1"], random_data["p1"])
