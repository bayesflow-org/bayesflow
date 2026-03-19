import keras
import pytest

from bayesflow.networks import UNet, UViT, ResidualUViT
from bayesflow.utils.serialization import serialize, deserialize
from tests.utils import assert_layers_equal

BATCH = 2
H, W, C = 8, 12, 3
COND_C = 2


def _inputs():
    x = keras.random.normal((BATCH, H, W, C))
    t = keras.random.uniform((BATCH,))
    cond = keras.random.normal((BATCH, H, W, COND_C))
    return x, t, cond


def _input_shapes():
    return (BATCH, H, W, C), (BATCH,), (BATCH, H, W, COND_C)


BACKBONE_CONFIGS = [
    (UNet, dict(widths=(8, 16), res_blocks=1, attn_stage=(False, False))),
    (UViT, dict(widths=(8, 16), res_blocks=1, transformer_blocks=1)),
    (ResidualUViT, dict(widths=(8, 16), res_blocks_down=1, res_blocks_up=1, transformer_blocks=1)),
]


@pytest.mark.parametrize("cls, kwargs", BACKBONE_CONFIGS, ids=lambda v: v.__name__ if isinstance(v, type) else "")
def test_output_shape(cls, kwargs):
    net = cls(**kwargs)
    x, t, cond = _inputs()
    y = net((x, t, cond), training=False)
    assert keras.ops.shape(y) == keras.ops.shape(x)


@pytest.mark.parametrize("cls, kwargs", BACKBONE_CONFIGS, ids=lambda v: v.__name__ if isinstance(v, type) else "")
def test_serialize_deserialize(cls, kwargs):
    net = cls(**kwargs)
    net.build(_input_shapes())

    serialized = serialize(net)
    deserialized = deserialize(serialized)
    reserialized = serialize(deserialized)

    assert keras.tree.lists_to_tuples(serialized) == keras.tree.lists_to_tuples(reserialized)


@pytest.mark.parametrize("cls, kwargs", BACKBONE_CONFIGS, ids=lambda v: v.__name__ if isinstance(v, type) else "")
def test_save_and_load(tmp_path, cls, kwargs):
    net = cls(**kwargs)
    net.build(_input_shapes())

    keras.saving.save_model(net, tmp_path / "model.keras")
    loaded = keras.saving.load_model(tmp_path / "model.keras")

    assert_layers_equal(net, loaded)
