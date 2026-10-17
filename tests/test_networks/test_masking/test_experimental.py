import keras
import numpy as np
import pytest

from bayesflow.experimental import FreeFormFlow, LatentInferenceNetwork
from bayesflow.experimental.graphical_approximator.non_exchangeable_wrapper import NonExchangeableWrapper
from bayesflow.networks import DeepSet, InferenceNetwork
from bayesflow.utils.serialization import deserialize, serialize


class RecordingSubnet(keras.Layer):
    """Minimal subnet exposing the semantic-mask contract for routing tests."""

    def __init__(self):
        super().__init__()
        self.fixed_target_mask = None
        self.infer_target_mask = None
        self.observed_condition_mask = None
        self.attention_mask = None
        self.mask = None

    def call(
        self,
        inputs,
        training=None,
        fixed_target_mask=None,
        infer_target_mask=None,
        observed_condition_mask=None,
        attention_mask=None,
        mask=None,
    ):
        self.fixed_target_mask = fixed_target_mask
        self.infer_target_mask = infer_target_mask
        self.observed_condition_mask = observed_condition_mask
        self.attention_mask = attention_mask
        self.mask = mask
        return inputs

    def compute_output_shape(self, input_shape):
        return input_shape


class StubAutoEncoder(keras.Layer):
    """Deterministic dimension-reducing autoencoder for wrapper contract tests."""

    def build(self, input_shape):
        self.output_dim = input_shape[-1]
        self.built = True

    def compute_output_shape(self, input_shape):
        return (*input_shape[:-1], 2)

    def call(self, x, training=False, inverse=False):
        del training
        if inverse:
            padding = [(0, 0)] * len(x.shape)
            padding[-1] = (0, self.output_dim - x.shape[-1])
            return keras.ops.pad(x, padding)
        return x[..., :2]

    def compute_metrics(self, x, sample_weight=None, stage="training", **kwargs):
        del sample_weight, stage, kwargs
        return {"loss": keras.ops.zeros((), dtype=x.dtype), "z": self(x)}


class RecordingInferenceNetwork(InferenceNetwork):
    def __init__(self):
        super().__init__()
        self.fixed_target_mask = None
        self.infer_target_mask = None
        self.observed_condition_mask = None
        self.fixed_target_value = None

    def build(self, xz_shape, conditions_shape=None):
        del conditions_shape
        self.base_distribution.build(xz_shape)
        self.built = True

    def _record(self, kwargs):
        self.fixed_target_mask = kwargs.get("fixed_target_mask")
        self.infer_target_mask = kwargs.get("infer_target_mask")
        self.observed_condition_mask = kwargs.get("observed_condition_mask")
        self.fixed_target_value = kwargs.get("fixed_target_value")

    def _forward(self, x, conditions=None, density=False, training=False, **kwargs):
        del conditions, training
        self._record(kwargs)
        if density:
            return x, keras.ops.zeros(keras.ops.shape(x)[:-1], dtype=x.dtype)
        return x

    def _inverse(self, z, conditions=None, density=False, training=False, **kwargs):
        del conditions, training
        self._record(kwargs)
        if self.fixed_target_mask is not None:
            z = self.fixed_target_mask * z + (1.0 - self.fixed_target_mask) * self.fixed_target_value
        if density:
            return z, keras.ops.zeros(keras.ops.shape(z)[:-1], dtype=z.dtype)
        return z

    def compute_metrics(
        self,
        x,
        conditions=None,
        sample_weight=None,
        stage="training",
        **kwargs,
    ):
        del conditions, sample_weight, stage
        self._record(kwargs)
        return {"loss": keras.ops.zeros((), dtype=x.dtype)}


def _small_free_form_flow(**kwargs):
    return FreeFormFlow(
        encoder_subnet_kwargs={"widths": (8,)},
        decoder_subnet_kwargs={"widths": (8,)},
        hutchinson_sampling="sphere",
        **kwargs,
    )


def test_generated_masks_are_resolved_once_and_routed(targets, conditions):
    encoder = RecordingSubnet()
    decoder = RecordingSubnet()
    network = FreeFormFlow(
        encoder_subnet=encoder,
        decoder_subnet=decoder,
        hutchinson_sampling="sphere",
        fixed_target_prob=1.0,
        missing_target_prob=1.0,
        missing_conditions_prob=1.0,
    )
    network.build(targets.shape, conditions.shape)

    metrics = network.compute_metrics(targets, conditions=conditions, stage="training")

    assert keras.ops.isfinite(metrics["loss"])
    for subnet in (encoder, decoder):
        assert keras.backend.standardize_dtype(subnet.fixed_target_mask.dtype) == "float32"
        assert keras.ops.all(keras.ops.sum(subnet.fixed_target_mask, axis=-1) == 1)
        assert keras.ops.all(subnet.infer_target_mask == subnet.fixed_target_mask)
        assert keras.ops.all(subnet.observed_condition_mask == 0)

    assert keras.ops.all(encoder.fixed_target_mask == decoder.fixed_target_mask)


def test_free_form_generates_missingness_only_when_encoder_supports_it(targets, conditions):
    class UnmaskedSubnet(keras.Layer):
        def call(self, inputs, training=None):
            del training
            return inputs

        def compute_output_shape(self, input_shape):
            return input_shape

    decoder = RecordingSubnet()
    network = FreeFormFlow(
        encoder_subnet=UnmaskedSubnet(),
        decoder_subnet=decoder,
        hutchinson_sampling="sphere",
        fixed_target_prob=1.0,
        missing_target_prob=1.0,
        missing_conditions_prob=1.0,
    )
    network.build(targets.shape, conditions.shape)

    metrics = network.compute_metrics(targets, conditions=conditions, stage="training")

    assert keras.ops.isfinite(metrics["loss"])
    assert decoder.fixed_target_mask is not None
    assert decoder.infer_target_mask is None
    assert decoder.observed_condition_mask is None


def test_explicit_masks_override_generation(targets, conditions):
    encoder = RecordingSubnet()
    decoder = RecordingSubnet()
    network = FreeFormFlow(
        encoder_subnet=encoder,
        decoder_subnet=decoder,
        hutchinson_sampling="sphere",
        fixed_target_prob=1.0,
        missing_target_prob=1.0,
        missing_conditions_prob=1.0,
    )
    network.build(targets.shape, conditions.shape)
    fixed = np.ones((3,), dtype=np.float32)
    inferred = np.ones((3,), dtype=np.float32)
    observed = np.ones((2,), dtype=np.float32)

    network.compute_metrics(
        targets,
        conditions=conditions,
        stage="training",
        fixed_target_mask=fixed,
        infer_target_mask=inferred,
        observed_condition_mask=observed,
    )

    for subnet in (encoder, decoder):
        np.testing.assert_array_equal(
            keras.ops.convert_to_numpy(subnet.fixed_target_mask),
            np.ones((2, 3), dtype=np.float32),
        )
        np.testing.assert_array_equal(
            keras.ops.convert_to_numpy(subnet.infer_target_mask),
            np.ones((2, 3), dtype=np.float32),
        )
        np.testing.assert_array_equal(
            keras.ops.convert_to_numpy(subnet.observed_condition_mask),
            np.ones((2, 2), dtype=np.float32),
        )


def test_ordinary_masks_are_routed_to_both_free_form_subnets(targets, conditions):
    encoder = RecordingSubnet()
    decoder = RecordingSubnet()
    network = FreeFormFlow(
        encoder_subnet=encoder,
        decoder_subnet=decoder,
        hutchinson_sampling="sphere",
    )
    network.build(targets.shape, conditions.shape)
    attention = keras.ops.ones((2, 3, 5), dtype="float32")
    padding = keras.ops.convert_to_tensor(
        np.asarray(
            [[1.0, 1.0, 0.0], [1.0, 0.0, 0.0]],
            dtype=np.float32,
        )
    )

    metrics = network.compute_metrics(
        targets,
        conditions=conditions,
        stage="validation",
        attention_mask=attention,
        mask=padding,
    )

    assert keras.ops.isfinite(metrics["loss"])
    for subnet in (encoder, decoder):
        assert keras.ops.all(subnet.attention_mask == attention)
        assert keras.ops.all(subnet.mask == padding)


def test_sample_clamps_fixed_targets_to_explicit_values():
    network = _small_free_form_flow()
    network.build((2, 3))
    fixed_target_mask = np.asarray([1.0, 0.0, 1.0], dtype=np.float32)
    fixed_target_value = np.asarray([0.0, 7.0, 0.0], dtype=np.float32)

    samples = network.sample(
        (2,),
        seed=1729,
        fixed_target_mask=fixed_target_mask,
        fixed_target_value=fixed_target_value,
    )

    np.testing.assert_array_equal(
        keras.ops.convert_to_numpy(samples[..., 1]),
        np.full((2,), 7.0, dtype=np.float32),
    )


@pytest.mark.skip_on_mps
def test_encode_and_density_use_explicit_fixed_target_values():
    network = _small_free_form_flow()
    network.build((2, 3))
    targets = keras.ops.convert_to_tensor(np.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32))
    fixed_target_mask = np.asarray([1.0, 0.0, 1.0], dtype=np.float32)
    fixed_target_value = np.asarray([0.0, 7.0, 0.0], dtype=np.float32)

    encoded = network.encode(
        targets,
        fixed_target_mask=fixed_target_mask,
        fixed_target_value=fixed_target_value,
    )
    _, log_density = network(
        targets,
        density=True,
        fixed_target_mask=fixed_target_mask,
        fixed_target_value=fixed_target_value,
    )
    changed_targets = keras.ops.convert_to_tensor(np.asarray([[1.0, 200.0, 3.0], [4.0, -500.0, 6.0]], dtype=np.float32))
    _, changed_log_density = network(
        changed_targets,
        density=True,
        fixed_target_mask=fixed_target_mask,
        fixed_target_value=fixed_target_value,
    )

    np.testing.assert_array_equal(
        keras.ops.convert_to_numpy(encoded[..., 1]),
        np.full((2,), 7.0, dtype=np.float32),
    )
    assert np.all(np.isfinite(keras.ops.convert_to_numpy(log_density)))
    np.testing.assert_allclose(
        keras.ops.convert_to_numpy(log_density),
        keras.ops.convert_to_numpy(changed_log_density),
        rtol=1e-6,
        atol=1e-6,
    )


@pytest.mark.skip_on_mps
def test_conditional_density_does_not_include_fixed_coordinates():
    network = _small_free_form_flow()
    network.build((2, 3))
    fixed_target_mask = np.asarray([1.0, 0.0, 1.0], dtype=np.float32)
    first = keras.ops.convert_to_tensor(np.asarray([[1.0, 2.0, 3.0], [0.5, -4.0, -1.0]], dtype=np.float32))
    second = keras.ops.convert_to_tensor(np.asarray([[1.0, 200.0, 3.0], [0.5, 400.0, -1.0]], dtype=np.float32))

    first_log_prob = network.log_prob(first, fixed_target_mask=fixed_target_mask)
    second_log_prob = network.log_prob(second, fixed_target_mask=fixed_target_mask)

    np.testing.assert_allclose(
        keras.ops.convert_to_numpy(first_log_prob),
        keras.ops.convert_to_numpy(second_log_prob),
        rtol=1e-6,
        atol=1e-6,
    )


def test_all_fixed_targets_have_zero_training_loss(targets):
    network = _small_free_form_flow(beta=10.0)
    network.build(targets.shape)

    metrics = network.compute_metrics(
        targets,
        stage="training",
        fixed_target_mask=np.zeros((3,), dtype=np.float32),
    )

    np.testing.assert_allclose(
        keras.ops.convert_to_numpy(metrics["loss"]),
        np.asarray(0.0, dtype=np.float32),
        rtol=0.0,
        atol=0.0,
    )


def test_mask_configuration_round_trips_through_serialization():
    network = _small_free_form_flow(
        fixed_target_prob=0.2,
        missing_target_prob=0.3,
        missing_conditions_prob=0.4,
        seed=91,
    )

    config = network.get_config()
    restored = deserialize(serialize(network))

    assert config["fixed_target_prob"] == pytest.approx(0.2)
    assert config["missing_target_prob"] == pytest.approx(0.3)
    assert config["missing_conditions_prob"] == pytest.approx(0.4)
    assert config["seed"] == 91
    assert restored.fixed_target_prob == pytest.approx(0.2)
    assert restored.missing_target_prob == pytest.approx(0.3)
    assert restored.missing_conditions_prob == pytest.approx(0.4)
    assert restored.seed == 91
    assert restored.mask_handler.seed_generator is restored.seed_generator


@pytest.mark.parametrize("probability", ["fixed_target_prob", "missing_target_prob"])
def test_latent_network_rejects_target_mask_probabilities(probability, targets):
    network = LatentInferenceNetwork(
        StubAutoEncoder(),
        RecordingInferenceNetwork(),
        **{probability: 0.5},
    )
    network.build(targets.shape)

    with pytest.raises(NotImplementedError, match="cannot project target masks"):
        network.compute_metrics(targets, stage="training")


@pytest.mark.parametrize("mask_name", ["fixed_target_mask", "infer_target_mask", "fixed_target_value"])
def test_latent_network_rejects_explicit_target_masks(mask_name, targets):
    network = LatentInferenceNetwork(StubAutoEncoder(), RecordingInferenceNetwork())
    network.build(targets.shape)
    value = np.ones((3,), dtype=np.float32)

    with pytest.raises(NotImplementedError, match="cannot project target masks"):
        network(targets, **{mask_name: value})


def test_latent_network_delegates_explicit_condition_mask(targets, conditions):
    inner = RecordingInferenceNetwork()
    network = LatentInferenceNetwork(StubAutoEncoder(), inner)
    network.build(targets.shape, conditions.shape)
    observed = np.asarray([1.0, 0.0], dtype=np.float32)

    network(targets, conditions=conditions, observed_condition_mask=observed)

    np.testing.assert_array_equal(
        keras.ops.convert_to_numpy(inner.observed_condition_mask),
        np.asarray([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32),
    )


def test_latent_network_generates_and_delegates_condition_mask(targets, conditions):
    inner = RecordingInferenceNetwork()
    network = LatentInferenceNetwork(
        StubAutoEncoder(),
        inner,
        missing_conditions_prob=1.0,
    )
    network.build(targets.shape, conditions.shape)

    metrics = network.compute_metrics(targets, conditions=conditions, stage="training")

    assert keras.ops.isfinite(metrics["loss"])
    np.testing.assert_array_equal(
        keras.ops.convert_to_numpy(inner.observed_condition_mask),
        np.zeros((2, 2), dtype=np.float32),
    )


def test_non_exchangeable_wrapper_flattens_and_routes_generated_masks():
    targets = keras.ops.convert_to_tensor(
        np.asarray(
            [
                [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
                [[-1.0, -2.0], [-3.0, -4.0], [-5.0, -6.0]],
            ],
            dtype=np.float32,
        )
    )
    conditions = keras.ops.convert_to_tensor(
        np.asarray(
            [[[0.5], [1.0], [1.5]], [[-0.5], [-1.0], [-1.5]]],
            dtype=np.float32,
        )
    )
    inner = RecordingInferenceNetwork()
    wrapper = NonExchangeableWrapper(
        inner,
        DeepSet(summary_dim=2),
        fixed_target_prob=1.0,
        missing_conditions_prob=1.0,
    )

    metrics = wrapper.compute_metrics(targets, conditions=conditions, stage="training")

    assert wrapper.built
    assert keras.ops.isfinite(metrics["loss"])
    assert inner.fixed_target_mask.shape == (6, 1, 2)
    assert keras.ops.all(keras.ops.sum(inner.fixed_target_mask, axis=-1) == 1)
    assert inner.infer_target_mask is None
    assert keras.ops.all(inner.observed_condition_mask[..., :2] == 1)
    assert keras.ops.all(inner.observed_condition_mask[..., 2:] == 0)


def test_non_exchangeable_wrapper_slices_fixed_values_during_sampling():
    inner = RecordingInferenceNetwork()
    wrapper = NonExchangeableWrapper(inner, DeepSet(summary_dim=2))
    wrapper.build((2, 3, 2))
    fixed_target_mask = np.asarray([1.0, 0.0], dtype=np.float32)
    fixed_target_value = np.asarray(
        [
            [[0.0, 10.0], [0.0, 20.0], [0.0, 30.0]],
            [[0.0, -10.0], [0.0, -20.0], [0.0, -30.0]],
        ],
        dtype=np.float32,
    )

    samples = wrapper.sample(
        (2, 3),
        seed=1729,
        fixed_target_mask=fixed_target_mask,
        fixed_target_value=fixed_target_value,
    )

    np.testing.assert_array_equal(
        keras.ops.convert_to_numpy(samples[..., 1]),
        fixed_target_value[..., 1],
    )


def test_non_exchangeable_wrapper_rejects_missing_target_probability():
    wrapper = NonExchangeableWrapper(
        RecordingInferenceNetwork(),
        DeepSet(summary_dim=2),
        missing_target_prob=0.5,
    )
    targets = keras.ops.convert_to_tensor(np.ones((2, 3, 2), dtype=np.float32))

    with pytest.raises(NotImplementedError, match="without an explicit feature-mask protocol"):
        wrapper.compute_metrics(targets, stage="training")


def test_non_exchangeable_wrapper_rejects_infer_target_mask():
    wrapper = NonExchangeableWrapper(RecordingInferenceNetwork(), DeepSet(summary_dim=2))
    wrapper.build((2, 3, 2))
    targets = keras.ops.convert_to_tensor(np.ones((2, 3, 2), dtype=np.float32))

    with pytest.raises(NotImplementedError, match="without an explicit feature-mask protocol"):
        wrapper(targets, infer_target_mask=np.ones((2,), dtype=np.float32))
