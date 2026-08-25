import numpy as np
import keras
import pytest

from bayesflow import AutoregressiveApproximator
from bayesflow.adapters import Adapter
from bayesflow.approximators.helpers import AutoregressiveConditionBuilder, AutoregressiveSampler
from bayesflow.networks import CouplingFlow, TimeSeriesTransformer
from bayesflow.networks.decoders import RecurrentDecoder, TransformerDecoder
from tests.utils import assert_models_equal


class NoOpStandardizer:
    def maybe_standardize(self, value, **kwargs):
        return value


class TimeAxisEncoder(keras.Layer):
    def __init__(self, time_axis: int):
        super().__init__()
        self.time_axis = time_axis

    def call(self, x, training=False, **kwargs):
        return x


class TimeEchoDecoder(keras.Layer):
    def call(self, inference_variables, encoder_outputs, *, time=None, **kwargs):
        if time is None:
            return keras.ops.zeros((*keras.ops.shape(inference_variables)[:-1], 1))
        return keras.ops.expand_dims(time, axis=-1)


class CacheTimeDecoder:
    def initialize_cache(self, encoder_outputs, encoder_mask=None, time=None):
        return {"time": time}

    def decode_step(self, previous_target, *, step, cache, **kwargs):
        return cache["time"][:, step : step + 1], cache


class EchoInferenceNetwork:
    def sample(self, batch_shape, conditions=None, seed=None, **kwargs):
        return conditions


@pytest.fixture(params=["recurrent", "deep_recurrent", "transformer"])
def autoregressive_approximator(request):
    if request.param == "recurrent":
        decoder = RecurrentDecoder(embed_dim=8)
    elif request.param == "deep_recurrent":
        decoder = RecurrentDecoder(embed_dim=(8, 6), recurrent_type=("gru", "lstm"))
    else:
        decoder = TransformerDecoder(
            embed_dim=8,
            output_dim=4,
            num_layers=1,
            num_heads=2,
            dropout=0.0,
            time_embed_dim=4,
        )

    return AutoregressiveApproximator(
        inference_network=CouplingFlow(
            depth=1,
            permutation=None,
            use_actnorm=False,
            subnet_kwargs={"widths": (8, 8)},
        ),
        encoder_network=TimeSeriesTransformer(
            summary_dim=4,
            embed_dims=(8,),
            num_heads=(2,),
            dropout=0.0,
            time_embed_dim=4,
            time_axis=2,
            return_sequences=True,
        ),
        decoder_network=decoder,
        standardize=None,
    )


@pytest.fixture
def autoregressive_data():
    rng = np.random.default_rng(123)
    batch_size, num_steps = 3, 5
    time = np.broadcast_to(np.arange(num_steps, dtype="float32"), (batch_size, num_steps))
    summary_variables = np.concatenate(
        [rng.normal(size=(batch_size, num_steps, 2)).astype("float32"), time[..., None]],
        axis=-1,
    )
    inference_mask = np.array(
        [
            [True, True, True, False, False],
            [True, True, True, True, False],
            [True, True, False, False, False],
        ]
    )
    return {
        "inference_variables": rng.normal(size=(batch_size, num_steps, 2)).astype("float32"),
        "summary_variables": summary_variables,
        "summary_mask": inference_mask.copy(),
        "inference_mask": inference_mask,
    }


def build_approximator(approximator, data):
    data_shapes = {key: value.shape for key, value in data.items()}
    approximator.build(data_shapes)


def test_condition_builder_uses_encoder_time_axis_for_decoder_time():
    summary_variables = keras.ops.convert_to_tensor(
        np.array(
            [
                [[0.0, 10.0, 1.0], [1.0, 11.0, 2.0], [2.0, 12.0, 3.0]],
                [[3.0, 13.0, 4.0], [4.0, 14.0, 5.0], [5.0, 15.0, 6.0]],
            ],
            dtype="float32",
        )
    )
    inference_variables = keras.ops.zeros((2, 3, 2))

    builder = AutoregressiveConditionBuilder()
    encoder_outputs, decoder_time = builder.resolve_encoder(
        standardizer=NoOpStandardizer(),
        encoder_network=TimeAxisEncoder(time_axis=1),
        inference_conditions=None,
        summary_variables=summary_variables,
        stage="inference",
    )
    conditions, _ = builder.resolve(
        standardizer=NoOpStandardizer(),
        encoder_network=TimeAxisEncoder(time_axis=1),
        decoder_network=TimeEchoDecoder(),
        inference_variables=inference_variables,
        inference_conditions=None,
        summary_variables=summary_variables,
        stage="inference",
    )

    expected_time = np.array([[10.0, 11.0, 12.0], [13.0, 14.0, 15.0]], dtype="float32")
    np.testing.assert_allclose(keras.ops.convert_to_numpy(decoder_time), expected_time)
    np.testing.assert_allclose(
        keras.ops.convert_to_numpy(encoder_outputs),
        keras.ops.convert_to_numpy(summary_variables),
    )
    np.testing.assert_allclose(keras.ops.convert_to_numpy(conditions[..., 0]), expected_time)


def test_autoregressive_sampler_repeats_time_for_each_sample():
    encoder_outputs = keras.ops.zeros((2, 3, 4))
    time = keras.ops.convert_to_tensor(
        np.array(
            [
                [0.0, 2.0, 5.0],
                [1.0, 3.0, 8.0],
            ],
            dtype="float32",
        )
    )

    samples = AutoregressiveSampler().sample(
        inference_network=EchoInferenceNetwork(),
        decoder_network=CacheTimeDecoder(),
        num_samples=2,
        conditions=encoder_outputs,
        time=time,
    )

    expected = np.repeat(keras.ops.convert_to_numpy(time)[:, None, :, None], repeats=2, axis=1)
    np.testing.assert_allclose(keras.ops.convert_to_numpy(samples), expected)


def test_autoregressive_sampler_uses_sample_shape_as_sequence_length():
    encoder_outputs = keras.ops.zeros((2, 3, 4))
    time = keras.ops.convert_to_tensor(
        np.array(
            [
                [0.0, 2.0, 5.0],
                [1.0, 3.0, 8.0],
            ],
            dtype="float32",
        )
    )

    samples = AutoregressiveSampler().sample(
        inference_network=EchoInferenceNetwork(),
        decoder_network=CacheTimeDecoder(),
        num_samples=2,
        conditions=encoder_outputs,
        sample_shape=2,
        time=time,
    )

    expected = np.repeat(keras.ops.convert_to_numpy(time[:, :2])[:, None, :, None], repeats=2, axis=1)
    np.testing.assert_allclose(keras.ops.convert_to_numpy(samples), expected)


def test_transformer_decoder_cached_decode_matches_teacher_forcing():
    batch_size, num_steps, target_dim, encoder_dim = 2, 4, 2, 3
    inference_variables = keras.random.normal((batch_size, num_steps, target_dim), seed=1)
    encoder_outputs = keras.random.normal((batch_size, num_steps, encoder_dim), seed=2)
    explicit_time = keras.ops.convert_to_tensor(
        np.array(
            [
                [0.0, 0.5, 1.5, 3.0],
                [0.0, 0.25, 1.0, 2.0],
            ],
            dtype="float32",
        )
    )

    for time in (None, explicit_time):
        decoder = TransformerDecoder(
            embed_dim=8,
            output_dim=5,
            num_layers=2,
            num_heads=2,
            dropout=0.0,
            time_embed_dim=4,
        )
        decoder.build(
            (batch_size, num_steps, target_dim),
            (batch_size, num_steps, encoder_dim),
        )

        teacher_forced = decoder(inference_variables, encoder_outputs, time=time, training=False)
        cache = decoder.initialize_cache(encoder_outputs, time=time)
        previous_target = None
        cached_conditions = []

        for step in range(num_steps):
            condition, cache = decoder.decode_step(previous_target, step=step, cache=cache)
            cached_conditions.append(condition)
            previous_target = inference_variables[:, step]

        cached_conditions = keras.ops.stack(cached_conditions, axis=1)
        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(cached_conditions),
            keras.ops.convert_to_numpy(teacher_forced),
            rtol=1e-5,
            atol=1e-5,
        )


@pytest.mark.parametrize(
    "embed_dim,recurrent_type",
    [
        (8, "gru"),
        (8, "lstm"),
        ((4, 5), "gru"),
        ((4, 5), ("gru", "lstm")),
    ],
)
def test_recurrent_decoder_cached_decode_matches_teacher_forcing(embed_dim, recurrent_type):
    batch_size, num_steps, target_dim, encoder_dim = 2, 4, 2, 3
    inference_variables = keras.random.normal((batch_size, num_steps, target_dim), seed=1)
    encoder_outputs = keras.random.normal((batch_size, num_steps, encoder_dim), seed=2)

    decoder = RecurrentDecoder(
        embed_dim=embed_dim,
        recurrent_type=recurrent_type,
    )
    decoder.build(
        (batch_size, num_steps, target_dim),
        (batch_size, num_steps, encoder_dim),
    )

    teacher_forced = decoder(inference_variables, encoder_outputs, training=False)
    cache = decoder.initialize_cache(encoder_outputs)
    previous_target = None
    cached_conditions = []

    for step in range(num_steps):
        condition, cache = decoder.decode_step(previous_target, step=step, cache=cache)
        cached_conditions.append(condition)
        previous_target = inference_variables[:, step]

    cached_conditions = keras.ops.stack(cached_conditions, axis=1)
    np.testing.assert_allclose(
        keras.ops.convert_to_numpy(cached_conditions),
        keras.ops.convert_to_numpy(teacher_forced),
        rtol=1e-5,
        atol=1e-5,
    )


def test_builds_all_autoregressive_networks(autoregressive_approximator, autoregressive_data):
    build_approximator(autoregressive_approximator, autoregressive_data)

    assert autoregressive_approximator.built
    assert autoregressive_approximator.encoder_network.built
    assert autoregressive_approximator.decoder_network.built
    assert autoregressive_approximator.inference_network.built


def test_sample_with_each_decoder(autoregressive_approximator, autoregressive_data):
    build_approximator(autoregressive_approximator, autoregressive_data)
    conditions = {key: value for key, value in autoregressive_data.items() if key != "inference_variables"}

    samples = autoregressive_approximator.sample(
        num_samples=2,
        conditions=conditions,
        batch_size=1,
        seed=123,
    )["inference_variables"]

    assert samples.shape == (3, 2, 5, 2)
    assert np.all(np.isfinite(samples))
    expanded_mask = autoregressive_data["inference_mask"][:, None, :, None]
    np.testing.assert_allclose(np.where(expanded_mask, 0.0, samples), 0.0, atol=0.0)


def test_log_prob_with_each_decoder(autoregressive_approximator, autoregressive_data):
    build_approximator(autoregressive_approximator, autoregressive_data)

    log_prob = autoregressive_approximator.log_prob(autoregressive_data)

    assert log_prob.shape == (3,)
    assert np.all(np.isfinite(log_prob))


def test_log_prob_adds_adapter_log_det_after_summing_steps(autoregressive_approximator, autoregressive_data):
    build_approximator(autoregressive_approximator, autoregressive_data)
    data = {
        "inference_variables": autoregressive_data["inference_variables"],
        "summary_variables": autoregressive_data["summary_variables"],
    }
    scale = np.float32(2.0)
    transformed_data = data | {"inference_variables": data["inference_variables"] * scale}
    expected = autoregressive_approximator.log_prob(transformed_data)

    autoregressive_approximator.adapter = Adapter().scale("inference_variables", by=scale)
    actual = autoregressive_approximator.log_prob(data)

    event_size = np.prod(data["inference_variables"].shape[1:])
    np.testing.assert_allclose(actual, expected + event_size * np.log(scale), rtol=1e-5, atol=1e-5)


def test_padding_masks_ignore_masked_values(autoregressive_approximator, autoregressive_data):
    build_approximator(autoregressive_approximator, autoregressive_data)
    expected = autoregressive_approximator.log_prob(autoregressive_data)

    perturbed = {key: value.copy() for key, value in autoregressive_data.items()}
    padding = ~autoregressive_data["inference_mask"]
    perturbed["inference_variables"][padding] = 1e4
    perturbed["summary_variables"][padding] = -1e4
    actual = autoregressive_approximator.log_prob(perturbed)

    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)


def test_save_and_load_with_each_decoder(tmp_path, autoregressive_approximator, autoregressive_data):
    build_approximator(autoregressive_approximator, autoregressive_data)
    expected = autoregressive_approximator.log_prob(autoregressive_data)
    model_path = tmp_path / "autoregressive.keras"

    keras.saving.save_model(autoregressive_approximator, model_path)
    loaded = keras.saving.load_model(model_path)
    actual = loaded.log_prob(autoregressive_data)

    assert type(loaded.decoder_network) is type(autoregressive_approximator.decoder_network)
    assert_models_equal(autoregressive_approximator, loaded)
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)
