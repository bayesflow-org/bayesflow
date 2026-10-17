import keras
import numpy as np
import pytest

from bayesflow.adapters import Adapter
from bayesflow.approximators import Approximator, AutoregressiveApproximator, EnsembleApproximator
from bayesflow.approximators.helpers import AutoregressiveSampler
from bayesflow.networks import InferenceNetwork


def _numpy(value):
    return keras.ops.convert_to_numpy(value)


class _CacheDecoder:
    def initialize_cache(self, encoder_outputs, **kwargs):
        return {"encoder_outputs": encoder_outputs}

    def decode_step(self, previous_target, *, step, cache, **kwargs):
        return cache["encoder_outputs"][:, step], cache


class _RecordingSamplerNetwork:
    def __init__(self, target_dim):
        self.target_dim = target_dim
        self.calls = []

    def sample(self, batch_shape, conditions=None, seed=None, **kwargs):
        self.calls.append({key: np.array(_numpy(value), copy=True) for key, value in kwargs.items()})
        return keras.ops.zeros((*batch_shape, self.target_dim), dtype="float32")


def _encoder_outputs(batch_size=3, num_steps=3):
    values = np.arange(batch_size * num_steps * 2, dtype=np.float32)
    return keras.ops.convert_to_tensor(values.reshape(batch_size, num_steps, 2))


def test_autoregressive_sampler_preserves_shared_vector_semantic_masks():
    network = _RecordingSamplerNetwork(target_dim=2)
    shared = {
        "fixed_target_mask": np.asarray([1.0, 0.0], dtype=np.float32),
        "fixed_target_value": np.asarray([4.0, 7.0], dtype=np.float32),
        "infer_target_mask": np.asarray([1.0, 1.0], dtype=np.float32),
        "observed_condition_mask": np.asarray([0.0, 1.0], dtype=np.float32),
    }

    samples = AutoregressiveSampler().sample(
        inference_network=network,
        decoder_network=_CacheDecoder(),
        num_samples=2,
        conditions=_encoder_outputs(),
        batch_size=1,
        seed=keras.random.SeedGenerator(101),
        **shared,
    )

    assert tuple(samples.shape) == (3, 2, 3, 2)
    assert len(network.calls) == 9
    for call in network.calls:
        for key, expected in shared.items():
            assert call[key].dtype == np.float32
            np.testing.assert_array_equal(call[key], expected)


def test_autoregressive_sampler_slices_shared_time_semantic_masks():
    network = _RecordingSamplerNetwork(target_dim=2)
    # Deliberately use B == T. A rank-two (T, D) semantic mask is shared,
    # not a batch tensor, even when its leading dimension happens to equal B.
    shared_time_mask = np.asarray(
        [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
        dtype=np.float32,
    )
    shared_time_value = np.asarray(
        [[2.0, 3.0], [4.0, 5.0], [6.0, 7.0]],
        dtype=np.float32,
    )
    shared_time_infer = np.ones((3, 2), dtype=np.float32)
    shared_time_observed = np.asarray(
        [[1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
        dtype=np.float32,
    )

    AutoregressiveSampler().sample(
        inference_network=network,
        decoder_network=_CacheDecoder(),
        num_samples=2,
        conditions=_encoder_outputs(batch_size=3, num_steps=3),
        batch_size=1,
        seed=keras.random.SeedGenerator(102),
        fixed_target_mask=shared_time_mask,
        fixed_target_value=shared_time_value,
        infer_target_mask=shared_time_infer,
        observed_condition_mask=shared_time_observed,
    )

    assert len(network.calls) == 9
    for batch_index in range(3):
        for step in range(3):
            call = network.calls[batch_index * 3 + step]
            np.testing.assert_array_equal(call["fixed_target_mask"], shared_time_mask[step])
            np.testing.assert_array_equal(call["fixed_target_value"], shared_time_value[step])
            np.testing.assert_array_equal(call["infer_target_mask"], shared_time_infer[step])
            np.testing.assert_array_equal(call["observed_condition_mask"], shared_time_observed[step])


def test_autoregressive_sampler_batches_repeats_and_slices_semantic_masks():
    network = _RecordingSamplerNetwork(target_dim=2)
    batch_size, num_steps, target_dim = 3, 3, 2
    fixed_mask = np.asarray(
        [
            [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
            [[0.0, 1.0], [1.0, 0.0], [0.0, 0.0]],
            [[1.0, 1.0], [1.0, 0.0], [0.0, 1.0]],
        ],
        dtype=np.float32,
    )
    fixed_value = np.arange(batch_size * num_steps * target_dim, dtype=np.float32).reshape(
        batch_size, num_steps, target_dim
    )
    infer_mask = np.ones((batch_size, num_steps, target_dim), dtype=np.float32)
    observed_mask = np.flip(fixed_mask, axis=-1).copy()
    direct_mask = np.asarray(
        [[1.0, 0.0, 1.0, 1.0], [0.0, 1.0, 1.0, 0.0], [1.0, 1.0, 0.0, 0.0]],
        dtype=np.float32,
    )
    direct_attention = np.stack(
        [np.eye(4, dtype=np.float32), np.ones((4, 4), dtype=np.float32), np.tri(4, dtype=np.float32)],
        axis=0,
    )

    AutoregressiveSampler().sample(
        inference_network=network,
        decoder_network=_CacheDecoder(),
        num_samples=2,
        conditions=_encoder_outputs(batch_size=batch_size, num_steps=num_steps),
        batch_size=2,
        seed=keras.random.SeedGenerator(103),
        fixed_target_mask=fixed_mask,
        fixed_target_value=fixed_value,
        infer_target_mask=infer_mask,
        observed_condition_mask=observed_mask,
        mask=direct_mask,
        attention_mask=direct_attention,
    )

    # The first sampler minibatch contains data rows 0 and 1, repeated twice.
    for step in range(num_steps):
        call = network.calls[step]
        expected_mask = np.repeat(fixed_mask[:2, step], repeats=2, axis=0)
        expected_value = np.repeat(fixed_value[:2, step], repeats=2, axis=0)
        np.testing.assert_array_equal(call["fixed_target_mask"], expected_mask)
        np.testing.assert_array_equal(call["fixed_target_value"], expected_value)
        np.testing.assert_array_equal(
            call["infer_target_mask"],
            np.repeat(infer_mask[:2, step], repeats=2, axis=0),
        )
        np.testing.assert_array_equal(
            call["observed_condition_mask"],
            np.repeat(observed_mask[:2, step], repeats=2, axis=0),
        )
        np.testing.assert_array_equal(call["mask"], np.repeat(direct_mask[:2], repeats=2, axis=0))
        np.testing.assert_array_equal(
            call["attention_mask"],
            np.repeat(direct_attention[:2], repeats=2, axis=0),
        )

    # The final minibatch contains data row 2 only, also repeated twice.
    for step in range(num_steps):
        call = network.calls[num_steps + step]
        np.testing.assert_array_equal(
            call["fixed_target_mask"],
            np.repeat(fixed_mask[2:3, step], repeats=2, axis=0),
        )
        np.testing.assert_array_equal(
            call["fixed_target_value"],
            np.repeat(fixed_value[2:3, step], repeats=2, axis=0),
        )
        np.testing.assert_array_equal(
            call["infer_target_mask"],
            np.repeat(infer_mask[2:3, step], repeats=2, axis=0),
        )
        np.testing.assert_array_equal(
            call["observed_condition_mask"],
            np.repeat(observed_mask[2:3, step], repeats=2, axis=0),
        )


def test_autoregressive_sampler_slices_shared_and_batched_singleton_time_semantic_masks():
    network = _RecordingSamplerNetwork(target_dim=2)
    batch_size, num_steps, num_samples = 3, 2, 2
    shared_fixed_mask = np.asarray([[1.0, 0.0]], dtype=np.float32)
    shared_fixed_value = np.asarray([[4.0, 7.0]], dtype=np.float32)
    batched_infer_mask = np.asarray(
        [
            [[1.0, 1.0]],
            [[1.0, 0.0]],
            [[0.0, 1.0]],
        ],
        dtype=np.float32,
    )
    batched_observed_mask = np.flip(batched_infer_mask, axis=-1).copy()

    AutoregressiveSampler().sample(
        inference_network=network,
        decoder_network=_CacheDecoder(),
        num_samples=num_samples,
        conditions=_encoder_outputs(batch_size=batch_size, num_steps=num_steps),
        batch_size=2,
        seed=keras.random.SeedGenerator(105),
        fixed_target_mask=shared_fixed_mask,
        fixed_target_value=shared_fixed_value,
        infer_target_mask=batched_infer_mask,
        observed_condition_mask=batched_observed_mask,
    )

    for batch_index, (start, stop) in enumerate(((0, 2), (2, 3))):
        expected_infer = np.repeat(batched_infer_mask[start:stop, 0], num_samples, axis=0)
        expected_observed = np.repeat(batched_observed_mask[start:stop, 0], num_samples, axis=0)
        for step in range(num_steps):
            call = network.calls[batch_index * num_steps + step]
            np.testing.assert_array_equal(call["fixed_target_mask"], shared_fixed_mask[0])
            np.testing.assert_array_equal(call["fixed_target_value"], shared_fixed_value[0])
            np.testing.assert_array_equal(call["infer_target_mask"], expected_infer)
            np.testing.assert_array_equal(call["observed_condition_mask"], expected_observed)


@pytest.mark.parametrize("time_size", [1, 2])
def test_autoregressive_sampler_slices_shared_time_indexed_ordinary_masks(time_size):
    network = _RecordingSamplerNetwork(target_dim=2)
    batch_size, num_steps = 3, 2
    mask = np.asarray(
        [[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]][:time_size],
        dtype=np.float32,
    )
    attention_mask = np.asarray(
        [
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        ][:time_size],
        dtype=np.float32,
    )

    AutoregressiveSampler().sample(
        inference_network=network,
        decoder_network=_CacheDecoder(),
        num_samples=2,
        conditions=_encoder_outputs(batch_size=batch_size, num_steps=num_steps),
        batch_size=2,
        seed=keras.random.SeedGenerator(106),
        mask=mask,
        attention_mask=attention_mask,
    )

    for batch_index in range(2):
        for step in range(num_steps):
            call = network.calls[batch_index * num_steps + step]
            time_index = 0 if time_size == 1 else step
            np.testing.assert_array_equal(call["mask"], mask[time_index])
            np.testing.assert_array_equal(call["attention_mask"], attention_mask[time_index])


@pytest.mark.parametrize("time_size", [1, 2])
def test_autoregressive_sampler_batches_repeats_and_slices_time_indexed_ordinary_masks(time_size):
    network = _RecordingSamplerNetwork(target_dim=2)
    batch_size, num_steps, num_samples = 3, 2, 2
    mask = np.asarray(
        [
            [[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]],
            [[0.0, 1.0, 0.0], [1.0, 0.0, 1.0]],
            [[1.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        ],
        dtype=np.float32,
    )[:, :time_size]
    attention_mask = np.asarray(
        [
            [
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            ],
            [
                [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]],
                [[1.0, 1.0, 0.0], [0.0, 1.0, 1.0]],
            ],
            [
                [[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]],
                [[0.0, 1.0, 1.0], [1.0, 0.0, 0.0]],
            ],
        ],
        dtype=np.float32,
    )[:, :time_size]

    AutoregressiveSampler().sample(
        inference_network=network,
        decoder_network=_CacheDecoder(),
        num_samples=num_samples,
        conditions=_encoder_outputs(batch_size=batch_size, num_steps=num_steps),
        batch_size=2,
        seed=keras.random.SeedGenerator(107),
        mask=mask,
        attention_mask=attention_mask,
    )

    for batch_index, (start, stop) in enumerate(((0, 2), (2, 3))):
        for step in range(num_steps):
            call = network.calls[batch_index * num_steps + step]
            time_index = 0 if time_size == 1 else step
            expected_mask = np.repeat(mask[start:stop, time_index], num_samples, axis=0)
            expected_attention = np.repeat(attention_mask[start:stop, time_index], num_samples, axis=0)
            np.testing.assert_array_equal(call["mask"], expected_mask)
            np.testing.assert_array_equal(call["attention_mask"], expected_attention)


class _IdentityEncoder(keras.Layer):
    def call(self, inputs, attention_mask=None, mask=None, training=None):
        return inputs


class _RecordingDecoder(keras.Layer):
    def __init__(self):
        super().__init__()
        self.received = None
        self.decode_received = []

    def call(
        self,
        inference_variables,
        encoder_outputs,
        *,
        target_mask=None,
        encoder_mask=None,
        attention_mask=None,
        training=None,
    ):
        self.received = {
            "target_mask": target_mask,
            "attention_mask": attention_mask,
        }
        return encoder_outputs

    def initialize_cache(self, encoder_outputs, **kwargs):
        return {"encoder_outputs": encoder_outputs}

    def decode_step(
        self,
        previous_target,
        *,
        step,
        cache,
        target_mask=None,
        attention_mask=None,
    ):
        self.decode_received.append(
            {
                "target_mask": target_mask,
                "attention_mask": attention_mask,
            }
        )
        return cache["encoder_outputs"][:, step], cache


class _RecordingInferenceNetwork(InferenceNetwork):
    def __init__(self):
        super().__init__()
        self.received = None
        self.sample_calls = []

    def _forward(self, x, conditions=None, density=False, training=False, **kwargs):
        if density:
            return x, keras.ops.zeros(keras.ops.shape(x)[:-1], dtype=x.dtype)
        return x

    def _inverse(self, z, conditions=None, density=False, training=False, **kwargs):
        if density:
            return z, keras.ops.zeros(keras.ops.shape(z)[:-1], dtype=z.dtype)
        return z

    def compute_metrics(self, x, conditions=None, sample_weight=None, stage="training", **kwargs):
        self.received = kwargs
        return {"loss": keras.ops.sum(x * keras.ops.cast(0.0, x.dtype))}

    def sample(self, batch_shape, conditions=None, seed=None, **kwargs):
        self.sample_calls.append(kwargs)
        return keras.ops.zeros((*batch_shape, 2), dtype="float32")


def test_autoregressive_approximator_keeps_sequence_and_subnet_masks_separate():
    inference_network = _RecordingInferenceNetwork()
    decoder = _RecordingDecoder()
    approximator = AutoregressiveApproximator(
        inference_network=inference_network,
        encoder_network=_IdentityEncoder(),
        decoder_network=decoder,
        standardize=None,
    )
    targets = keras.ops.ones((2, 3, 2), dtype="float32")
    summaries = keras.ops.ones((2, 3, 2), dtype="float32")
    sequence_mask = keras.ops.convert_to_tensor(np.asarray([[1.0, 1.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32))
    sequence_attention = keras.ops.ones((2, 3, 3), dtype="float32")
    direct_mask = keras.ops.convert_to_tensor(np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32))
    direct_attention = keras.ops.convert_to_tensor(
        np.asarray(
            [
                [[1.0, 0.0], [0.0, 1.0]],
                [[0.0, 1.0], [1.0, 0.0]],
            ],
            dtype=np.float32,
        )
    )
    fixed_target_mask = keras.ops.ones_like(targets)

    approximator.compute_metrics(
        targets,
        summary_variables=summaries,
        inference_mask=sequence_mask,
        inference_attention_mask=sequence_attention,
        mask=direct_mask,
        attention_mask=direct_attention,
        fixed_target_mask=fixed_target_mask,
    )

    np.testing.assert_array_equal(_numpy(decoder.received["target_mask"]), _numpy(sequence_mask))
    np.testing.assert_array_equal(_numpy(decoder.received["attention_mask"]), _numpy(sequence_attention))
    np.testing.assert_array_equal(_numpy(inference_network.received["mask"]), _numpy(direct_mask))
    np.testing.assert_array_equal(
        _numpy(inference_network.received["attention_mask"]),
        _numpy(direct_attention),
    )
    np.testing.assert_array_equal(
        _numpy(inference_network.received["fixed_target_mask"]),
        _numpy(fixed_target_mask),
    )


def test_autoregressive_approximator_rejects_missing_targets_without_decoder_feature_masks():
    approximator = AutoregressiveApproximator(
        inference_network=_RecordingInferenceNetwork(),
        encoder_network=_IdentityEncoder(),
        decoder_network=_RecordingDecoder(),
        standardize=None,
    )
    targets = keras.ops.ones((2, 3, 2), dtype="float32")
    summaries = keras.ops.ones((2, 3, 2), dtype="float32")

    with pytest.raises(NotImplementedError, match="feature-mask-aware decoder"):
        approximator.compute_metrics(
            targets,
            summary_variables=summaries,
            infer_target_mask=keras.ops.ones_like(targets),
        )


def test_autoregressive_sampling_gives_explicit_sequence_masks_precedence():
    inference_network = _RecordingInferenceNetwork()
    decoder = _RecordingDecoder()
    approximator = AutoregressiveApproximator(
        inference_network=inference_network,
        encoder_network=_IdentityEncoder(),
        decoder_network=decoder,
        standardize=None,
    )
    summaries = np.ones((2, 3, 2), dtype=np.float32)
    discovered_mask = np.ones((2, 3), dtype=np.float32)
    discovered_attention = np.ones((2, 3, 3), dtype=np.float32)
    explicit_mask = np.asarray([[1.0, 0.0, 0.0], [1.0, 1.0, 0.0]], dtype=np.float32)
    explicit_attention = np.stack(
        [np.eye(3, dtype=np.float32), np.tri(3, dtype=np.float32)],
        axis=0,
    )
    direct_mask = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    fixed_target_mask = np.ones((2, 3, 2), dtype=np.float32)
    discovered_fixed_value = np.full((2, 3, 2), -1.0, dtype=np.float32)
    explicit_fixed_value = np.arange(12, dtype=np.float32).reshape(2, 3, 2)

    approximator.sample(
        num_samples=1,
        conditions={
            "summary_variables": summaries,
            "inference_mask": discovered_mask,
            "inference_attention_mask": discovered_attention,
            "fixed_target_mask": fixed_target_mask,
            "fixed_target_value": discovered_fixed_value,
        },
        seed=keras.random.SeedGenerator(104),
        to_numpy=False,
        inference_mask=explicit_mask,
        inference_attention_mask=explicit_attention,
        mask=direct_mask,
        fixed_target_value=explicit_fixed_value,
    )

    assert len(decoder.decode_received) == 3
    for call in decoder.decode_received:
        np.testing.assert_array_equal(_numpy(call["target_mask"]), explicit_mask)
        np.testing.assert_array_equal(_numpy(call["attention_mask"]), explicit_attention)
    assert len(inference_network.sample_calls) == 3
    for step, call in enumerate(inference_network.sample_calls):
        np.testing.assert_array_equal(_numpy(call["mask"]), direct_mask)
        np.testing.assert_array_equal(_numpy(call["fixed_target_mask"]), fixed_target_mask[:, step])
        np.testing.assert_array_equal(_numpy(call["fixed_target_value"]), explicit_fixed_value[:, step])
        assert "inference_mask" not in call
        assert "inference_attention_mask" not in call


def test_autoregressive_log_prob_combines_sequence_and_fixed_coordinate_masks():
    approximator = AutoregressiveApproximator(
        inference_network=_RecordingInferenceNetwork(),
        encoder_network=_IdentityEncoder(),
        decoder_network=_RecordingDecoder(),
        adapter=Adapter(),
        standardize="inference_variables",
    )
    targets = keras.ops.convert_to_tensor(
        np.asarray(
            [
                [[1.0, 2.0], [3.0, 6.0], [5.0, 10.0]],
                [[2.0, 4.0], [4.0, 8.0], [6.0, 12.0]],
            ],
            dtype=np.float32,
        )
    )
    sequence_mask = keras.ops.convert_to_tensor(np.asarray([[1.0, 1.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32))
    fixed_target_mask = keras.ops.convert_to_tensor(
        np.asarray(
            [
                [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
                [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]],
            ],
            dtype=np.float32,
        )
    )
    approximator.standardizer.build({"inference_variables": tuple(targets.shape)})
    approximator.standardizer.maybe_standardize(
        targets,
        key="inference_variables",
        stage="training",
        mask=sequence_mask,
    )

    actual = approximator.log_prob(
        {
            "inference_variables": _numpy(targets),
            "summary_variables": _numpy(targets),
        },
        to_numpy=False,
        inference_mask=sequence_mask,
        fixed_target_mask=fixed_target_mask,
    )

    std = approximator.standardizer.standardize_layers["inference_variables"].moving_std(0)
    step_log_det = -keras.ops.sum(keras.ops.log(std) * fixed_target_mask, axis=-1)
    expected = keras.ops.sum(step_log_det * sequence_mask, axis=-1)
    np.testing.assert_allclose(_numpy(actual), _numpy(expected), rtol=1e-6, atol=1e-6)


class _RecordingApproximator(Approximator):
    def __init__(self):
        super().__init__()
        self.adapter = Adapter()
        self.received = None

    def compute_metrics(self, inference_variables, stage="training", **kwargs):
        self.received = {"inference_variables": inference_variables, **kwargs}
        return {"loss": keras.ops.convert_to_tensor(np.float32(1.0))}


class _RecordingDistributionApproximator(_RecordingApproximator):
    has_distribution = True

    def __init__(self):
        super().__init__()
        self.sample_received = None
        self.log_prob_received = None

    def sample(self, *, num_samples, conditions, **kwargs):
        self.sample_received = kwargs
        batch_size = next(iter(conditions.values())).shape[0]
        return {
            "inference_variables": np.zeros(
                (batch_size, num_samples, 1),
                dtype=np.float32,
            )
        }

    def log_prob(self, data, **kwargs):
        self.log_prob_received = kwargs
        batch_size = next(iter(data.values())).shape[0]
        return np.zeros((batch_size,), dtype=np.float32)


@pytest.mark.parametrize("stage", ["training", "validation"])
def test_ensemble_routes_member_specific_semantic_masks_at_every_stage(stage):
    left = _RecordingApproximator()
    right = _RecordingApproximator()
    ensemble = EnsembleApproximator({"left": left, "right": right})
    values = {
        "left": keras.ops.ones((2, 2), dtype="float32"),
        "right": keras.ops.zeros((2, 2), dtype="float32"),
    }
    semantic = {
        "fixed_target_mask": values,
        "fixed_target_value": {
            "left": keras.ops.full((2, 2), 3.0, dtype="float32"),
            "right": keras.ops.full((2, 2), 7.0, dtype="float32"),
        },
        "infer_target_mask": values,
        "observed_condition_mask": {
            "left": keras.ops.ones((2, 1), dtype="float32"),
            "right": keras.ops.zeros((2, 1), dtype="float32"),
        },
    }

    metrics = ensemble.compute_metrics(
        inference_variables=values,
        stage=stage,
        **semantic,
    )

    assert np.isclose(_numpy(metrics["loss"]), np.float32(2.0))
    for name, member in (("left", left), ("right", right)):
        np.testing.assert_array_equal(_numpy(member.received["inference_variables"]), _numpy(values[name]))
        for key, member_values in semantic.items():
            received = _numpy(member.received[key])
            assert received.dtype == np.float32
            np.testing.assert_array_equal(received, _numpy(member_values[name]))


def test_ensemble_broadcasts_shared_semantic_masks_to_all_members():
    left = _RecordingApproximator()
    right = _RecordingApproximator()
    ensemble = EnsembleApproximator({"left": left, "right": right})
    targets = keras.ops.ones((2, 2), dtype="float32")
    shared_mask = keras.ops.convert_to_tensor(np.asarray([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32))

    ensemble.compute_metrics(
        inference_variables=targets,
        fixed_target_mask=shared_mask,
    )

    np.testing.assert_array_equal(_numpy(left.received["fixed_target_mask"]), _numpy(shared_mask))
    np.testing.assert_array_equal(_numpy(right.received["fixed_target_mask"]), _numpy(shared_mask))


def test_ensemble_routes_member_specific_semantic_masks_during_inference():
    left = _RecordingDistributionApproximator()
    right = _RecordingDistributionApproximator()
    ensemble = EnsembleApproximator({"left": left, "right": right})
    data = {"inference_conditions": np.ones((2, 1), dtype=np.float32)}
    semantic = {
        "fixed_target_mask": {
            "left": np.ones((2, 2), dtype=np.float32),
            "right": np.zeros((2, 2), dtype=np.float32),
        },
        "fixed_target_value": {
            "left": np.full((2, 2), np.float32(3.0), dtype=np.float32),
            "right": np.full((2, 2), np.float32(7.0), dtype=np.float32),
        },
        "infer_target_mask": {
            "left": np.ones((2, 2), dtype=np.float32),
            "right": np.zeros((2, 2), dtype=np.float32),
        },
        "observed_condition_mask": {
            "left": np.ones((2, 1), dtype=np.float32),
            "right": np.zeros((2, 1), dtype=np.float32),
        },
    }

    ensemble.sample(
        num_samples=2,
        conditions=data,
        merge_members=False,
        **semantic,
    )
    ensemble.log_prob(
        data=data,
        merge_members=False,
        **semantic,
    )

    for name, member in (("left", left), ("right", right)):
        for key, values in semantic.items():
            sampled = _numpy(member.sample_received[key])
            evaluated = _numpy(member.log_prob_received[key])
            assert sampled.dtype == np.float32
            assert evaluated.dtype == np.float32
            np.testing.assert_array_equal(sampled, values[name])
            np.testing.assert_array_equal(evaluated, values[name])
