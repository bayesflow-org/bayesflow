import numpy as np
import keras

from bayesflow.networks.standardization import Standardization
from bayesflow.utils.serialization import serialize, deserialize

from tests.utils import assert_layers_equal


def test_forward_standardization_training():
    random_input = keras.random.normal((8, 4))

    layer = Standardization(momentum=0.0)  # no EMA for test stability
    layer.build(random_input.shape)

    out = layer(random_input, stage="training")

    moving_mean = keras.ops.convert_to_numpy(layer.moving_mean)
    moving_std = keras.ops.convert_to_numpy(layer.moving_std)
    random_input = keras.ops.convert_to_numpy(random_input)
    out = keras.ops.convert_to_numpy(out)

    # mean should now match the batch input
    np.testing.assert_allclose(moving_mean, np.mean(random_input, axis=0), atol=1e-5)
    np.testing.assert_allclose(moving_std, np.std(random_input, axis=0), atol=1e-5)

    assert out.shape == random_input.shape
    assert not np.any(np.isnan(out))


def test_inverse_standardization_ldj():
    random_input = keras.random.normal((1, 3))

    layer = Standardization(momentum=0.0)
    layer.build(random_input.shape)

    _ = layer(random_input, stage="training", forward=True)  # trigger moment update
    inv_x, ldj = layer(random_input, stage="inference", forward=False, log_det_jac=True)

    assert inv_x.shape == random_input.shape
    assert ldj.shape == random_input.shape[:-1]


def test_consistency_forward_inverse():
    random_input = keras.random.normal((4, 20, 5))
    layer = Standardization(momentum=0.0)
    layer.build((5,))
    standardized = layer(random_input, stage="training", forward=True)
    recovered = layer(standardized, stage="inference", forward=False)

    random_input = keras.ops.convert_to_numpy(random_input)
    recovered = keras.ops.convert_to_numpy(recovered)

    np.testing.assert_allclose(random_input, recovered, atol=1e-4)


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
