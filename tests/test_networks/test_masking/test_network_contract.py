import keras
import numpy as np
import pytest

from bayesflow.utils.serialization import deserialize, serialize

from .conftest import make_network


def _numpy(value):
    return keras.ops.convert_to_numpy(value)


def _build(network, targets, conditions):
    network.build(tuple(targets.shape), conditions_shape=tuple(conditions.shape))
    return network


def _skip_unsupported_torch_forward_ad(network_name, subnet):
    if network_name not in ("stable_consistency", "mean_flow") or subnet != "diffusion_transformer":
        return
    if keras.backend.backend() != "torch":
        return

    import torch

    if torch.cuda.device_count() == 0:
        pytest.skip(f"PyTorch CPU attention does not support the forward-mode AD used by {network_name}.")


class RecordingSubnet(keras.Layer):
    """Minimal time subnet exposing every supported masking argument explicitly."""

    def __init__(self):
        super().__init__()
        self.received = {}

    def call(
        self,
        inputs,
        training=None,
        attention_mask=None,
        mask=None,
        fixed_target_mask=None,
        infer_target_mask=None,
        observed_condition_mask=None,
        **kwargs,
    ):
        self.received = {
            "state": inputs[0],
            "attention_mask": attention_mask,
            "mask": mask,
            "fixed_target_mask": fixed_target_mask,
            "infer_target_mask": infer_target_mask,
            "observed_condition_mask": observed_condition_mask,
        }
        return inputs[0]

    def compute_output_shape(self, input_shape):
        return input_shape[0]


def test_network_owns_handler_configuration(network_factory):
    network = network_factory(
        fixed_target_prob=0.2,
        missing_target_prob=0.3,
        missing_conditions_prob=0.4,
        seed=19,
    )

    assert network.mask_handler is not None
    assert network.mask_handler.seed_generator is network.seed_generator
    assert network.fixed_target_prob == pytest.approx(0.2)
    assert network.missing_target_prob == pytest.approx(0.3)
    assert network.missing_conditions_prob == pytest.approx(0.4)

    config = network.get_config()
    assert config["fixed_target_prob"] == pytest.approx(0.2)
    assert config["missing_target_prob"] == pytest.approx(0.3)
    assert config["missing_conditions_prob"] == pytest.approx(0.4)
    assert config["seed"] == 19


def test_mask_probabilities_remain_settable_for_custom_network_compatibility(network_factory):
    network = network_factory()

    network.fixed_target_prob = 0.2
    network.missing_target_prob = 0.3
    network.missing_conditions_prob = 0.4

    assert network.mask_handler.fixed_target_prob == pytest.approx(0.2)
    assert network.mask_handler.missing_target_prob == pytest.approx(0.3)
    assert network.mask_handler.missing_conditions_prob == pytest.approx(0.4)


def test_mask_configuration_survives_serialization(
    network_factory,
    targets,
    conditions,
):
    network = _build(
        network_factory(
            fixed_target_prob=0.2,
            missing_target_prob=0.3,
            missing_conditions_prob=0.4,
            seed=19,
        ),
        targets,
        conditions,
    )

    restored = deserialize(serialize(network))

    assert restored.fixed_target_prob == pytest.approx(0.2)
    assert restored.missing_target_prob == pytest.approx(0.3)
    assert restored.missing_conditions_prob == pytest.approx(0.4)
    assert restored.seed == 19
    assert restored.mask_handler.seed_generator is restored.seed_generator


@pytest.mark.parametrize("subnet", ["time_mlp", "diffusion_transformer"])
def test_probability_generated_masking_has_finite_training_loss(
    network_name,
    network_factory,
    subnet,
    targets,
    conditions,
):
    _skip_unsupported_torch_forward_ad(network_name, subnet)
    network = _build(
        network_factory(
            subnet=subnet,
            fixed_target_prob=0.5,
            missing_target_prob=0.5,
            missing_conditions_prob=0.5,
        ),
        targets,
        conditions,
    )

    metrics = network.compute_metrics(targets, conditions=conditions, stage="training")

    loss = _numpy(metrics["loss"])
    assert loss.dtype == np.float32
    assert np.all(np.isfinite(loss))


def test_validation_never_samples_semantic_masks(
    monkeypatch,
    network_factory,
    targets,
    conditions,
):
    network = _build(
        network_factory(
            fixed_target_prob=1.0,
            missing_target_prob=1.0,
            missing_conditions_prob=1.0,
        ),
        targets,
        conditions,
    )

    def fail(*args, **kwargs):
        raise AssertionError("validation must not generate semantic masks")

    monkeypatch.setattr(network.mask_handler, "random_mask", fail)

    metrics = network.compute_metrics(targets, conditions=conditions, stage="validation")

    assert np.all(np.isfinite(_numpy(metrics["loss"])))


@pytest.mark.parametrize(
    ("probabilities", "expected_masks"),
    [
        ((0.0, 0.0, 0.0), None),
        ((1.0, 1.0, 1.0), "generated"),
    ],
    ids=["none", "all"],
)
def test_generated_masks_follow_probability_extremes_and_reach_subnet(
    network_factory,
    targets,
    conditions,
    probabilities,
    expected_masks,
):
    subnet = RecordingSubnet()
    network = _build(
        network_factory(
            subnet=subnet,
            fixed_target_prob=probabilities[0],
            missing_target_prob=probabilities[1],
            missing_conditions_prob=probabilities[2],
        ),
        targets,
        conditions,
    )

    network.compute_metrics(targets, conditions=conditions, stage="training")

    if expected_masks is None:
        assert subnet.received["fixed_target_mask"] is None
        assert subnet.received["infer_target_mask"] is None
        assert subnet.received["observed_condition_mask"] is None
        return

    fixed = _numpy(subnet.received["fixed_target_mask"])
    inferred = _numpy(subnet.received["infer_target_mask"])
    observed = _numpy(subnet.received["observed_condition_mask"])
    np.testing.assert_array_equal(fixed.sum(axis=-1), np.ones((2,), dtype=np.float32))
    np.testing.assert_array_equal(inferred, fixed)
    np.testing.assert_array_equal(observed, np.zeros((2, 2), dtype=np.float32))


def test_explicit_masks_override_sampling_and_reach_explicit_subnet_parameters(
    network_factory,
    targets,
    conditions,
    fixed_target_mask,
    infer_target_mask,
    observed_condition_mask,
):
    subnet = RecordingSubnet()
    network = _build(
        network_factory(
            subnet=subnet,
            fixed_target_prob=1.0,
            missing_target_prob=1.0,
            missing_conditions_prob=1.0,
        ),
        targets,
        conditions,
    )
    attention_mask = keras.ops.ones((2, 3, 5), dtype="float32")
    padding_mask = keras.ops.ones((2, 3), dtype="float32")

    metrics = network.compute_metrics(
        targets,
        conditions=conditions,
        stage="training",
        attention_mask=attention_mask,
        mask=padding_mask,
        fixed_target_mask=fixed_target_mask,
        infer_target_mask=infer_target_mask,
        observed_condition_mask=observed_condition_mask,
    )

    assert np.isfinite(_numpy(metrics["loss"]))
    np.testing.assert_array_equal(_numpy(subnet.received["attention_mask"]), _numpy(attention_mask))
    np.testing.assert_array_equal(_numpy(subnet.received["mask"]), _numpy(padding_mask))
    np.testing.assert_array_equal(_numpy(subnet.received["fixed_target_mask"]), _numpy(fixed_target_mask))
    np.testing.assert_array_equal(_numpy(subnet.received["infer_target_mask"]), _numpy(infer_target_mask))
    np.testing.assert_array_equal(
        _numpy(subnet.received["observed_condition_mask"]),
        _numpy(observed_condition_mask),
    )


def test_network_raises_if_its_subnet_cannot_take_the_mask(network_factory, targets, conditions):
    network = _build(network_factory(), targets, conditions)

    with pytest.raises(ValueError, match="`mask` would be ignored"):
        network.compute_metrics(targets, conditions=conditions, mask=keras.ops.ones((2, 3)))


@pytest.mark.parametrize("name", ["attention_mask", "mask"])
def test_coupling_flow_raises_on_mask(name, targets, conditions):
    from bayesflow.networks import CouplingFlow

    network = _build(CouplingFlow(depth=1, subnet_kwargs={"widths": (8,)}), targets, conditions)
    mask = keras.ops.ones((2, 3), dtype="float32")

    with pytest.raises(ValueError, match=f"`{name}` would be ignored"):
        network.log_prob(targets, conditions=conditions, **{name: mask})
    with pytest.raises(ValueError, match=f"`{name}` would be ignored"):
        network.sample(2, conditions=conditions, **{name: mask})


def test_sampling_routes_explicit_semantic_masks_once_to_subnet(
    monkeypatch,
    network_factory,
    targets,
    conditions,
    infer_target_mask,
    observed_condition_mask,
):
    subnet = RecordingSubnet()
    network = _build(network_factory(subnet=subnet), targets, conditions)
    original_resolve = network.mask_handler.resolve
    calls = 0

    def counting_resolve(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original_resolve(*args, **kwargs)

    monkeypatch.setattr(network.mask_handler, "resolve", counting_resolve)

    samples = network.sample(
        (2,),
        conditions=conditions,
        seed=keras.random.SeedGenerator(317),
        infer_target_mask=infer_target_mask,
        observed_condition_mask=observed_condition_mask,
    )

    assert tuple(samples.shape) == tuple(targets.shape)
    assert calls == 1
    np.testing.assert_array_equal(_numpy(subnet.received["infer_target_mask"]), _numpy(infer_target_mask))
    np.testing.assert_array_equal(
        _numpy(subnet.received["observed_condition_mask"]),
        _numpy(observed_condition_mask),
    )


def test_all_fixed_targets_have_zero_training_loss(network_factory, targets, conditions):
    network = _build(network_factory(), targets, conditions)
    fixed_target_mask = keras.ops.zeros_like(targets)

    metrics = network.compute_metrics(
        targets,
        conditions=conditions,
        stage="training",
        fixed_target_mask=fixed_target_mask,
    )

    np.testing.assert_allclose(_numpy(metrics["loss"]), np.asarray(0.0, dtype=np.float32), atol=1e-6)


@pytest.mark.parametrize("subnet", ["time_mlp", "diffusion_transformer"])
def test_sampling_keeps_fixed_targets_exactly(
    network_name,
    network_factory,
    subnet,
    targets,
    conditions,
    fixed_target_mask,
):
    _skip_unsupported_torch_forward_ad(network_name, subnet)
    network = _build(network_factory(subnet=subnet), targets, conditions)
    fixed_target_value = keras.ops.convert_to_tensor(
        np.asarray(
            [
                [-4.0, 7.5, 2.0],
                [3.0, -2.5, 1.0],
            ],
            dtype=np.float32,
        )
    )

    samples = network.sample(
        (2,),
        conditions=conditions,
        seed=keras.random.SeedGenerator(314),
        fixed_target_mask=fixed_target_mask,
        fixed_target_value=fixed_target_value,
    )

    assert _numpy(samples).dtype == np.float32
    np.testing.assert_allclose(
        _numpy(samples)[:, 1],
        _numpy(fixed_target_value)[:, 1],
        rtol=0.0,
        atol=1e-6,
    )


@pytest.mark.parametrize("network_name", ["flow_matching", "mean_flow"])
def test_flow_forward_uses_explicit_fixed_target_value(
    network_name,
    targets,
    conditions,
    fixed_target_mask,
):
    network = _build(make_network(network_name, subnet=RecordingSubnet()), targets, conditions)
    fixed_target_value = keras.ops.convert_to_tensor(
        np.asarray(
            [[-4.0, 7.5, 2.0], [3.0, -2.5, 1.0]],
            dtype=np.float32,
        )
    )

    encoded = network(
        targets,
        conditions=conditions,
        fixed_target_mask=fixed_target_mask,
        fixed_target_value=fixed_target_value,
    )

    np.testing.assert_allclose(
        _numpy(encoded)[:, 1],
        _numpy(fixed_target_value)[:, 1],
        rtol=0.0,
        atol=1e-6,
    )


# a call option of each network that differs from its default in make_network
CALL_OPTIONS = {
    "diffusion": dict(steps=3),
    "flow_matching": dict(steps=3),
    "consistency": dict(steps=4),
    "stable_consistency": dict(steps=3, rho=3.0),
    "mean_flow": dict(steps=3, rho=3.0, sigma_min=1e-2),
}


def test_call_options_are_settable_as_attributes_with_masks(
    network_name, network_factory, targets, conditions, fixed_target_mask
):
    """Keras forbids mixing tensor masks with non-tensor options in a call, so the options must be settable."""
    network = _build(network_factory(), targets, conditions)
    options = CALL_OPTIONS[network_name]
    z = keras.random.normal(keras.ops.shape(targets), seed=5)

    def sample(method, **kwargs):
        seed = keras.random.SeedGenerator(7)
        return _numpy(method(z, conditions=conditions, seed=seed, fixed_target_mask=fixed_target_mask, **kwargs))

    default = sample(network, inverse=True)
    expected = sample(network._inverse, **options)

    if network_name in ("diffusion", "flow_matching"):
        network.integrate_kwargs = network.integrate_kwargs | options
    else:
        for key, value in options.items():
            setattr(network, key, value)
    result = sample(network, inverse=True)

    np.testing.assert_allclose(result, expected, atol=1e-6)
    assert np.max(np.abs(result - default)) > 1e-6
