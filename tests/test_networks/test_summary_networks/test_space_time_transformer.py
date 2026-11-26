import numpy as np
import keras
import pytest

from bayesflow.networks import SpaceTimeTransformer

from .conftest import check_serialize_deserialize


BATCH = 2
NUM_STEPS = 5
HEIGHT = 9
WIDTH = 11
CHANNELS = 3
SUMMARY_DIM = 6


@pytest.fixture(autouse=True)
def random_seed():
    # The shared fixture sets a global seed, which shape checks do not need.
    return None


def _make(return_sequences=False, **kwargs):
    config = {
        "summary_dim": SUMMARY_DIM,
        "embed_dims": (8, 12),
        "num_heads": (2, 3),
        "patch_size": 4,
        "dropout": 0.0,
        "flash_attention": False,
        "return_sequences": return_sequences,
    }
    config.update(kwargs)
    return SpaceTimeTransformer(**config)


def _input(
    batch_size=BATCH,
    num_steps=NUM_STEPS,
    height=HEIGHT,
    width=WIDTH,
    channels=CHANNELS,
):
    return keras.ops.ones((batch_size, num_steps, height, width, channels))


def _masking_input():
    shape = (BATCH, NUM_STEPS, 8, 8, 1)
    values = keras.ops.arange(np.prod(shape), dtype="float32")
    return keras.ops.reshape(values, shape) / np.prod(shape)


def _frame_mask():
    return keras.ops.convert_to_tensor(
        [
            [True, True, False, True, False],
            [True, False, True, False, True],
        ],
        dtype="bool",
    )


def _replace_masked_frames_with_nan(x, mask):
    valid = mask[..., None, None, None]
    return keras.ops.where(valid, x, keras.ops.full_like(x, np.nan))


def test_default_output_shape():
    net = SpaceTimeTransformer()
    y = net(_input(batch_size=1, num_steps=2, height=4, width=4, channels=1), training=False)

    assert keras.ops.shape(y) == (1, 16)


def test_summary_output_shape():
    net = _make()
    y = net(_input(), training=False)

    assert keras.ops.shape(y) == (BATCH, SUMMARY_DIM)


def test_sequence_output_shape_preserves_num_steps():
    num_steps = 7
    net = _make(return_sequences=True)
    y = net(_input(num_steps=num_steps), training=False)

    assert keras.ops.shape(y) == (BATCH, num_steps, SUMMARY_DIM)


def test_odd_non_square_spatial_dimensions():
    net = _make(return_sequences=True)
    y = net(_input(height=7, width=10), training=False)

    assert keras.ops.shape(y) == (BATCH, NUM_STEPS, SUMMARY_DIM)


@pytest.mark.parametrize("time_embedding", [None, "time2vec", "lstm", "gru"])
def test_time_embedding_output_shape(time_embedding):
    net = _make(time_embedding=time_embedding, time_embed_dim=4)
    y = net(_input(), training=False)

    assert keras.ops.shape(y) == (BATCH, SUMMARY_DIM)


def test_symbolic_shape_preserves_dynamic_steps_and_resolution():
    net = _make(return_sequences=True)
    x = keras.Input(shape=(None, None, None, CHANNELS))
    model = keras.Model(x, net(x))

    assert model.output_shape == (None, None, SUMMARY_DIM)
    assert model(_input()).shape == (BATCH, NUM_STEPS, SUMMARY_DIM)
    assert model(_input(batch_size=1, num_steps=3, height=8, width=7)).shape == (1, 3, SUMMARY_DIM)


@pytest.mark.parametrize("mask_name", ["mask", "attention_mask"])
@pytest.mark.parametrize("return_sequences", [False, True])
def test_sequence_mask_shape(mask_name, return_sequences):
    net = _make(return_sequences=return_sequences)
    mask = keras.ops.ones((BATCH, NUM_STEPS), dtype="bool")
    y = net(_input(), training=False, **{mask_name: mask})

    expected = (BATCH, NUM_STEPS, SUMMARY_DIM) if return_sequences else (BATCH, SUMMARY_DIM)
    assert keras.ops.shape(y) == expected


def test_masked_nan_frames_do_not_change_summary():
    net = _make(embed_dims=(4,), num_heads=(1,), time_embedding=None)
    x = _masking_input()
    mask = _frame_mask()

    expected = keras.ops.convert_to_numpy(net(x, training=False, mask=mask))
    actual = keras.ops.convert_to_numpy(net(_replace_masked_frames_with_nan(x, mask), training=False, mask=mask))

    assert np.isfinite(actual).all()
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)


def test_attention_mask_takes_precedence_and_zeros_masked_sequence_outputs():
    net = _make(return_sequences=True, embed_dims=(4,), num_heads=(1,), time_embedding=None)
    x = _masking_input()
    attention_mask = _frame_mask()
    ignored_mask = keras.ops.ones_like(attention_mask, dtype="bool")

    expected = keras.ops.convert_to_numpy(net(x, training=False, attention_mask=attention_mask))
    actual = keras.ops.convert_to_numpy(
        net(
            _replace_masked_frames_with_nan(x, attention_mask),
            training=False,
            mask=ignored_mask,
            attention_mask=attention_mask,
        )
    )
    valid = keras.ops.convert_to_numpy(attention_mask)

    assert np.isfinite(actual).all()
    np.testing.assert_allclose(actual[valid], expected[valid], rtol=1e-6, atol=1e-6)
    np.testing.assert_array_equal(actual[~valid], 0.0)


@pytest.mark.parametrize("return_sequences", [False, True])
def test_masked_padding_matches_shorter_sequence(return_sequences):
    net = _make(return_sequences=return_sequences, embed_dims=(4,), num_heads=(1,), time_embedding=None)
    x = _masking_input()
    num_valid = 3
    mask = keras.ops.broadcast_to(keras.ops.arange(NUM_STEPS) < num_valid, (BATCH, NUM_STEPS))

    expected = net(x[:, :num_valid], training=False)
    actual = net(x, training=False, mask=mask)
    if return_sequences:
        actual = actual[:, :num_valid]

    np.testing.assert_allclose(keras.ops.convert_to_numpy(actual), keras.ops.convert_to_numpy(expected), atol=1e-6)


@pytest.mark.parametrize(
    ("return_sequences", "expected"),
    [
        (False, (BATCH, SUMMARY_DIM)),
        (True, (BATCH, NUM_STEPS, SUMMARY_DIM)),
    ],
)
def test_compute_output_shape(return_sequences, expected):
    net = _make(return_sequences=return_sequences)

    assert tuple(net.compute_output_shape((BATCH, NUM_STEPS, HEIGHT, WIDTH, CHANNELS))) == expected


def test_config_round_trip_with_non_default_options():
    net = _make(
        return_sequences=True,
        patch_size=3,
        dropout=0.1,
        expansion_factor=3.0,
        glu_variant="geglu",
        kernel_initializer="glorot_uniform",
        use_bias=True,
        layer_norm=False,
        gate_attention=True,
        gate_ffn=False,
        time_embedding="lstm",
        time_embed_dim=5,
        temporal_kernel_size=1,
    )

    check_serialize_deserialize(net, _input(height=8, width=10))


@pytest.mark.parametrize("return_sequences", [False, True])
def test_save_and_load_with_non_default_options(tmp_path, return_sequences):
    net = _make(
        return_sequences=return_sequences,
        patch_size=3,
        dropout=0.1,
        expansion_factor=3.0,
        time_embedding="gru",
        time_embed_dim=5,
        temporal_kernel_size=5,
        base_distribution=None if return_sequences else "normal",
    )
    model = keras.Sequential([keras.Input(shape=(None, None, None, CHANNELS)), net])
    x = keras.random.normal((BATCH, NUM_STEPS, 8, 10, CHANNELS), seed=97)
    expected = model(x, training=False)

    model_path = tmp_path / "space_time.keras"
    model.save(model_path)
    restored = keras.saving.load_model(model_path)

    assert restored.layers[0].patch_embedding.kernel_size == (5, 4, 4)
    assert type(restored.layers[0].base_distribution) is type(net.base_distribution)
    np.testing.assert_allclose(
        keras.ops.convert_to_numpy(restored(x, training=False)), keras.ops.convert_to_numpy(expected), atol=1e-6
    )
