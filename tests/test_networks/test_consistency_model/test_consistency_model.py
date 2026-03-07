import keras
import numpy as np
import pytest

from bayesflow.utils.serialization import serialize, deserialize
from tests.utils import assert_layers_equal


# ===========================================================================
# ConsistencyModel
# ===========================================================================


# ---- Build -----------------------------------------------------------------


def test_build(consistency_model, random_samples, random_conditions):
    xz_shape = keras.ops.shape(random_samples)
    cond_shape = keras.ops.shape(random_conditions) if random_conditions is not None else None

    assert not consistency_model.built
    consistency_model.build(xz_shape, conditions_shape=cond_shape)
    assert consistency_model.built
    assert consistency_model.variables


# ---- Forward raises --------------------------------------------------------


def test_forward_raises(consistency_model, random_samples, random_conditions):
    xz_shape = keras.ops.shape(random_samples)
    cond_shape = keras.ops.shape(random_conditions) if random_conditions is not None else None
    consistency_model.build(xz_shape, conditions_shape=cond_shape)

    with pytest.raises(NotImplementedError):
        consistency_model(random_samples, conditions=random_conditions)


# ---- Output shapes ---------------------------------------------------------


def test_inverse_output_shape(consistency_model, random_samples, random_conditions):
    xz_shape = keras.ops.shape(random_samples)
    cond_shape = keras.ops.shape(random_conditions) if random_conditions is not None else None
    consistency_model.build(xz_shape, conditions_shape=cond_shape)

    z = keras.random.normal(keras.ops.shape(random_samples))
    x = consistency_model(z, conditions=random_conditions, inverse=True)
    assert keras.ops.shape(x) == keras.ops.shape(random_samples)


# ---- Variable batch size ---------------------------------------------------


def test_variable_batch_size(consistency_model, random_samples, random_conditions):
    xz_shape = keras.ops.shape(random_samples)
    cond_shape = keras.ops.shape(random_conditions) if random_conditions is not None else None
    consistency_model.build(xz_shape, conditions_shape=cond_shape)

    for bs in [1, 4, 7]:
        z = keras.random.normal((bs,) + keras.ops.shape(random_samples)[1:])
        cond = (
            None if random_conditions is None else keras.random.normal((bs,) + keras.ops.shape(random_conditions)[1:])
        )
        out = consistency_model(z, conditions=cond, inverse=True)
        assert keras.ops.shape(out)[0] == bs


# ---- Serialization ---------------------------------------------------------


def test_serialize_deserialize(consistency_model, random_samples, random_conditions):
    xz_shape = keras.ops.shape(random_samples)
    cond_shape = keras.ops.shape(random_conditions) if random_conditions is not None else None
    consistency_model.build(xz_shape, conditions_shape=cond_shape)

    serialized = serialize(consistency_model)
    deserialized = deserialize(serialized)
    reserialized = serialize(deserialized)

    assert keras.tree.lists_to_tuples(serialized) == keras.tree.lists_to_tuples(reserialized)


def test_save_and_load(tmp_path, consistency_model, random_samples, random_conditions):
    xz_shape = keras.ops.shape(random_samples)
    cond_shape = keras.ops.shape(random_conditions) if random_conditions is not None else None
    consistency_model.build(xz_shape, conditions_shape=cond_shape)

    path = tmp_path / "consistency.keras"
    keras.saving.save_model(consistency_model, path)
    loaded = keras.saving.load_model(path)

    assert_layers_equal(consistency_model, loaded)


# ---- compute_metrics -------------------------------------------------------


def test_compute_metrics(consistency_model, random_samples, random_conditions):
    xz_shape = keras.ops.shape(random_samples)
    cond_shape = keras.ops.shape(random_conditions) if random_conditions is not None else None
    consistency_model.build(xz_shape, conditions_shape=cond_shape)

    metrics = consistency_model.compute_metrics(random_samples, conditions=random_conditions)
    assert "loss" in metrics
    loss = keras.ops.convert_to_numpy(metrics["loss"])
    assert np.isfinite(loss), f"Loss is not finite: {loss}"


def test_compute_metrics_with_masking(consistency_model_with_masking, random_samples, random_conditions):
    xz_shape = keras.ops.shape(random_samples)
    cond_shape = keras.ops.shape(random_conditions) if random_conditions is not None else None
    consistency_model_with_masking.build(xz_shape, conditions_shape=cond_shape)

    metrics = consistency_model_with_masking.compute_metrics(random_samples, conditions=random_conditions)
    assert "loss" in metrics
    loss = keras.ops.convert_to_numpy(metrics["loss"])
    assert np.isfinite(loss), f"Loss is not finite: {loss}"


# ===========================================================================
# StableConsistencyModel
# ===========================================================================


# ---- Build -----------------------------------------------------------------


def test_stable_build(stable_consistency_model, random_samples, random_conditions):
    xz_shape = keras.ops.shape(random_samples)
    cond_shape = keras.ops.shape(random_conditions) if random_conditions is not None else None

    assert not stable_consistency_model.built
    stable_consistency_model.build(xz_shape, conditions_shape=cond_shape)
    assert stable_consistency_model.built
    assert stable_consistency_model.variables


# ---- Forward raises --------------------------------------------------------


def test_stable_forward_raises(stable_consistency_model, random_samples, random_conditions):
    xz_shape = keras.ops.shape(random_samples)
    cond_shape = keras.ops.shape(random_conditions) if random_conditions is not None else None
    stable_consistency_model.build(xz_shape, conditions_shape=cond_shape)

    with pytest.raises(NotImplementedError):
        stable_consistency_model(random_samples, conditions=random_conditions)


# ---- Output shapes ---------------------------------------------------------


def test_stable_inverse_output_shape(stable_consistency_model, random_samples, random_conditions):
    xz_shape = keras.ops.shape(random_samples)
    cond_shape = keras.ops.shape(random_conditions) if random_conditions is not None else None
    stable_consistency_model.build(xz_shape, conditions_shape=cond_shape)

    z = keras.random.normal(keras.ops.shape(random_samples))
    x = stable_consistency_model(z, conditions=random_conditions, inverse=True)
    assert keras.ops.shape(x) == keras.ops.shape(random_samples)


# ---- Variable batch size ---------------------------------------------------


def test_stable_variable_batch_size(stable_consistency_model, random_samples, random_conditions):
    xz_shape = keras.ops.shape(random_samples)
    cond_shape = keras.ops.shape(random_conditions) if random_conditions is not None else None
    stable_consistency_model.build(xz_shape, conditions_shape=cond_shape)

    for bs in [1, 4, 7]:
        z = keras.random.normal((bs,) + keras.ops.shape(random_samples)[1:])
        cond = (
            None if random_conditions is None else keras.random.normal((bs,) + keras.ops.shape(random_conditions)[1:])
        )
        out = stable_consistency_model(z, conditions=cond, inverse=True)
        assert keras.ops.shape(out)[0] == bs


# ---- Serialization ---------------------------------------------------------


def test_stable_serialize_deserialize(stable_consistency_model, random_samples, random_conditions):
    xz_shape = keras.ops.shape(random_samples)
    cond_shape = keras.ops.shape(random_conditions) if random_conditions is not None else None
    stable_consistency_model.build(xz_shape, conditions_shape=cond_shape)

    serialized = serialize(stable_consistency_model)
    deserialized = deserialize(serialized)
    reserialized = serialize(deserialized)

    assert keras.tree.lists_to_tuples(serialized) == keras.tree.lists_to_tuples(reserialized)


def test_stable_save_and_load(tmp_path, stable_consistency_model, random_samples, random_conditions):
    xz_shape = keras.ops.shape(random_samples)
    cond_shape = keras.ops.shape(random_conditions) if random_conditions is not None else None
    stable_consistency_model.build(xz_shape, conditions_shape=cond_shape)

    path = tmp_path / "stable_consistency.keras"
    keras.saving.save_model(stable_consistency_model, path)
    loaded = keras.saving.load_model(path)

    assert_layers_equal(stable_consistency_model, loaded)


# ---- compute_metrics -------------------------------------------------------


def test_stable_compute_metrics(stable_consistency_model, random_samples, random_conditions):
    xz_shape = keras.ops.shape(random_samples)
    cond_shape = keras.ops.shape(random_conditions) if random_conditions is not None else None
    stable_consistency_model.build(xz_shape, conditions_shape=cond_shape)

    metrics = stable_consistency_model.compute_metrics(random_samples, conditions=random_conditions)
    assert "loss" in metrics
    loss = keras.ops.convert_to_numpy(metrics["loss"])
    assert np.isfinite(loss), f"Loss is not finite: {loss}"


def test_stable_compute_metrics_with_masking(stable_consistency_model_with_masking, random_samples, random_conditions):
    xz_shape = keras.ops.shape(random_samples)
    cond_shape = keras.ops.shape(random_conditions) if random_conditions is not None else None
    stable_consistency_model_with_masking.build(xz_shape, conditions_shape=cond_shape)

    metrics = stable_consistency_model_with_masking.compute_metrics(random_samples, conditions=random_conditions)
    assert "loss" in metrics
    loss = keras.ops.convert_to_numpy(metrics["loss"])
    assert np.isfinite(loss), f"Loss is not finite: {loss}"
