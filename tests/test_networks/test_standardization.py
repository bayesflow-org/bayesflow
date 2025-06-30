import numpy as np
import keras

from bayesflow.networks.standardization import Standardization
from bayesflow.utils.serialization import serialize, deserialize

from tests.utils import assert_layers_equal


def test_forward_standardization_training():
    random_input = keras.random.normal((8, 4))

    layer = Standardization()
    layer.build(random_input.shape)

    out = layer(random_input, stage="training")

    moving_mean = keras.ops.convert_to_numpy(layer.moving_mean[0])
    random_input = keras.ops.convert_to_numpy(random_input)
    out = keras.ops.convert_to_numpy(out)

    np.testing.assert_allclose(moving_mean, np.mean(random_input, axis=0), atol=1e-5)

    assert out.shape == random_input.shape
    assert not np.any(np.isnan(out))
    np.testing.assert_allclose(np.std(out, axis=0), 1.0, atol=1e-5)


def test_forward_standardization_training_constant_batch():
    constant_input = keras.ops.ones((8, 4))

    layer = Standardization()
    layer.build(constant_input.shape)

    out = layer(constant_input, stage="training")

    moving_mean = keras.ops.convert_to_numpy(layer.moving_mean[0])
    constant_input = keras.ops.convert_to_numpy(constant_input)
    out = keras.ops.convert_to_numpy(out)

    np.testing.assert_allclose(moving_mean, np.mean(constant_input, axis=0), atol=1e-5)

    assert out.shape == constant_input.shape
    assert not np.any(np.isnan(out))
    np.testing.assert_allclose(out, 0.0, atol=1e-5)
    np.testing.assert_allclose(np.std(out, axis=0), 0.0, atol=1e-5)


def test_inverse_standardization_ldj():
    random_input = keras.random.normal((1, 3))

    layer = Standardization(momentum=0.0)
    layer.build(random_input.shape)

    _ = layer(random_input, stage="training", forward=True)
    inv_x, ldj = layer(random_input, stage="inference", forward=False, log_det_jac=True)

    assert inv_x.shape == random_input.shape
    assert ldj.shape == random_input.shape[:-1]


def test_consistency_forward_inverse():
    random_input = keras.random.normal((4, 20, 5))
    layer = Standardization()
    _ = layer(random_input, stage="training", forward=True)

    standardized = layer(random_input, stage="inference", forward=True)
    recovered = layer(standardized, stage="inference", forward=False)

    random_input = keras.ops.convert_to_numpy(random_input)
    recovered = keras.ops.convert_to_numpy(recovered)

    np.testing.assert_allclose(random_input, recovered, atol=1e-4)


def test_nested_consistency_forward_inverse():
    random_input_a = keras.random.normal((2, 3, 5))
    random_input_b = keras.random.normal((4, 3))
    random_input = {"a": random_input_a, "b": random_input_b}

    layer = Standardization()

    _ = layer(random_input, stage="training", forward=True)
    standardized = layer(random_input, stage="inference", forward=True)
    recovered = layer(standardized, stage="inference", forward=False)

    random_input = keras.tree.map_structure(keras.ops.convert_to_numpy, random_input)
    recovered = keras.tree.map_structure(keras.ops.convert_to_numpy, recovered)

    np.testing.assert_allclose(random_input["a"], recovered["a"], atol=1e-4)
    np.testing.assert_allclose(random_input["b"], recovered["b"], atol=1e-4)


def test_nested_accuracy_forward():
    from bayesflow.utils import tree_concatenate

    # create inputs for two training passes
    random_input_a_1 = keras.random.normal((2, 3, 5))
    random_input_b_1 = keras.random.normal((4, 3))
    random_input_1 = {"a": random_input_a_1, "b": random_input_b_1}

    random_input_a_2 = keras.random.normal((3, 3, 5))
    random_input_b_2 = keras.random.normal((3, 3))
    random_input_2 = {"a": random_input_a_2, "b": random_input_b_2}

    # complete data for testing mean and std are 0 and 1
    random_input = tree_concatenate([random_input_1, random_input_2], axis=0)

    layer = Standardization()

    _ = layer(random_input_1, stage="training", forward=True)
    _ = layer(random_input_2, stage="training", forward=True)

    standardized = layer(random_input, stage="inference", forward=True)
    standardized = keras.tree.map_structure(keras.ops.convert_to_numpy, standardized)

    np.testing.assert_allclose(
        np.mean(standardized["a"], axis=tuple(range(standardized["a"].ndim - 1))), 0.0, atol=1e-4
    )
    np.testing.assert_allclose(
        np.mean(standardized["b"], axis=tuple(range(standardized["b"].ndim - 1))), 0.0, atol=1e-4
    )
    np.testing.assert_allclose(np.std(standardized["a"], axis=tuple(range(standardized["a"].ndim - 1))), 1.0, atol=1e-4)
    np.testing.assert_allclose(np.std(standardized["b"], axis=tuple(range(standardized["b"].ndim - 1))), 1.0, atol=1e-4)


def test_transformation_type_both_sides_scale():
    # Fix a known covariance and mean in original (not standardized space)
    covariance = np.array([[1, 0.5], [0.5, 2.0]], dtype="float32")
    mean = np.array([1, 10], dtype="float32")

    # Generate samples
    cholesky = keras.ops.cholesky(covariance)  # (dim, dim)
    normals = keras.random.normal((128, 2))  # (batch_size, dim)
    scaled = keras.ops.einsum("ij,bj->bi", cholesky, normals)

    random_input = keras.ops.convert_to_tensor(mean[None, :]) + scaled

    layer = Standardization()
    _ = layer(random_input, stage="training", forward=True)

    # Standardize samples
    standardized = layer(random_input, stage="inference", forward=True)
    # Compute covariance matrix in standardized space
    cov_standardized = np.cov(keras.ops.convert_to_numpy(standardized), rowvar=False)
    cov_standardized = keras.ops.convert_to_tensor(cov_standardized)
    # Inverse standardization of covariance matrix in standardized space
    cov_standardized_and_recovered = layer(
        cov_standardized, stage="inference", forward=False, transformation_type="both_sides_scale"
    )

    random_input = keras.ops.convert_to_numpy(random_input)
    cov_standardized_and_recovered = keras.ops.convert_to_numpy(cov_standardized_and_recovered)
    cov_input = np.cov(random_input, rowvar=False)

    np.testing.assert_allclose(cov_input, cov_standardized_and_recovered, atol=1e-4)


def test_transformation_type_left_side_scale():
    # Fix a known covariance and mean in original (not standardized space)
    covariance = np.array([[1, 0.5], [0.5, 2.0]], dtype="float32")
    mean = np.array([1, 10], dtype="float32")

    # Generate samples
    cholesky = keras.ops.cholesky(covariance)  # (dim, dim)
    normals = keras.random.normal((1024, 2))  # (batch_size, dim)
    scaled = keras.ops.einsum("ij,bj->bi", cholesky, normals)

    random_input = keras.ops.convert_to_tensor(mean[None, :]) + scaled

    layer = Standardization()
    _ = layer(random_input, stage="training", forward=True)

    # Standardize samples
    standardized = layer(random_input, stage="inference", forward=True)
    # Compute covariance matrix in standardized space
    cov_standardized = np.cov(keras.ops.convert_to_numpy(standardized), rowvar=False)
    cov_standardized = keras.ops.convert_to_tensor(cov_standardized)
    chol_standardized = keras.ops.cholesky(cov_standardized)  # (dim, dim)
    # Inverse standardization of covariance matrix in standardized space
    chol_standardized_and_recovered = layer(
        chol_standardized, stage="inference", forward=False, transformation_type="left_side_scale"
    )

    random_input = keras.ops.convert_to_numpy(random_input)
    chol_standardized_and_recovered = keras.ops.convert_to_numpy(chol_standardized_and_recovered)
    cov_input = np.cov(random_input, rowvar=False)
    chol_input = np.linalg.cholesky(cov_input)

    np.testing.assert_allclose(chol_input, chol_standardized_and_recovered, atol=1e-4)


def test_serialize_deserialize():
    layer = Standardization(momentum=0.0)
    layer.build((32, 5))

    serialized = serialize(layer)
    deserialized = deserialize(serialized)
    reserialized = serialize(deserialized)

    assert keras.tree.lists_to_tuples(serialized) == keras.tree.lists_to_tuples(reserialized)


def test_save_and_load(tmp_path):
    layer = Standardization(momentum=0.0)
    layer.build((32, 5))

    keras.saving.save_model(layer, tmp_path / "model.keras")
    loaded = keras.saving.load_model(tmp_path / "model.keras")

    assert_layers_equal(layer, loaded)
