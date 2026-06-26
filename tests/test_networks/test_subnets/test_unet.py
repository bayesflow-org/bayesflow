import keras
import pytest

from bayesflow.networks import ResidualUViT, UNet, UViT

from tests.utils import assert_layers_equal


UNET_CONFIGS = [
    (UNet, dict(widths=(8, 16), res_blocks=1, attn_stage=(False, False))),
    (UViT, dict(widths=(8, 16), res_blocks=1, transformer_blocks=1)),
    (ResidualUViT, dict(widths=(8, 16), res_blocks_down=1, res_blocks_up=1, transformer_blocks=1)),
]


def _tiny_inputs(batch_size: int = 2):
    x = keras.random.normal((batch_size, 9, 11, 3))
    t = keras.random.uniform((batch_size,))
    cond = keras.random.normal((batch_size, 9, 11, 2))
    return x, t, cond


@pytest.mark.parametrize("network_cls, kwargs", UNET_CONFIGS)
def test_output_shape(network_cls, kwargs):
    network = network_cls(**kwargs)
    x, t, cond = _tiny_inputs()
    y = network((x, t, cond), training=False)
    assert keras.ops.shape(y) == keras.ops.shape(x)


@pytest.mark.parametrize("network_cls, kwargs", UNET_CONFIGS)
def test_save_and_load(tmp_path, network_cls, kwargs):
    network = network_cls(**kwargs)
    # build by running a forward pass
    x, t, cond = _tiny_inputs()
    network((x, t, cond), training=False)

    keras.saving.save_model(network, tmp_path / "model.keras")
    loaded = keras.saving.load_model(tmp_path / "model.keras")

    assert_layers_equal(network, loaded)
