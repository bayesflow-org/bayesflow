import numpy as np
import keras

from bayesflow.approximators.helpers import AutoregressiveConditionBuilder, AutoregressiveSampler
from bayesflow.networks import TransformerDecoder


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

    conditions, _, decoder_time = AutoregressiveConditionBuilder().resolve(
        standardizer=NoOpStandardizer(),
        encoder_network=TimeAxisEncoder(time_axis=1),
        decoder_network=TimeEchoDecoder(),
        inference_variables=inference_variables,
        inference_conditions=None,
        summary_variables=summary_variables,
        stage="inference",
        return_decoder_time=True,
    )

    expected_time = np.array([[10.0, 11.0, 12.0], [13.0, 14.0, 15.0]], dtype="float32")
    np.testing.assert_allclose(keras.ops.convert_to_numpy(decoder_time), expected_time)
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
        num_steps=3,
        time=time,
    )

    expected = np.repeat(keras.ops.convert_to_numpy(time)[:, None, :, None], repeats=2, axis=1)
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
            summary_dim=5,
            embed_dim=8,
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
