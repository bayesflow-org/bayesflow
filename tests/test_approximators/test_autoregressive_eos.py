import keras
import numpy as np
import pytest

from bayesflow import AutoregressiveApproximator
from bayesflow.adapters import Adapter
from bayesflow.networks import CouplingFlow, TimeSeriesTransformer
from bayesflow.networks.decoders import RecurrentDecoder, TransformerDecoder


@pytest.fixture
def eos_data():
    rng = np.random.default_rng(5)
    values = np.full((3, 4, 2), -999.0, dtype="float32")
    mask = np.arange(4)[None, :] < np.array([0, 1, 2])[:, None]
    values[mask] = rng.normal(size=(mask.sum(), 2))
    return {
        "inference_variables": values,
        "inference_conditions": rng.normal(size=(3, 3)).astype("float32"),
        "inference_mask": mask,
    }


@pytest.fixture
def eos_approximator(eos_data):
    model = AutoregressiveApproximator(
        inference_network=CouplingFlow(depth=1, permutation=None, use_actnorm=False, subnet_kwargs={"widths": (8, 8)}),
        encoder_network=keras.layers.Identity(),
        decoder_network=TransformerDecoder(
            embed_dim=8,
            num_heads=2,
            num_layers=1,
            time_embed_dim=4,
            include_condition=False,
            dropout=0.0,
        ),
        standardize=None,
        eos_value=-999.0,
    )
    model.build({key: value.shape for key, value in eos_data.items()})
    return model


def metrics(model, data, **kwargs):
    return model.compute_metrics(
        **keras.tree.map_structure(keras.ops.convert_to_tensor, data), stage="validation", **kwargs
    )


def set_stop_logit(model, logit):
    model.eos_head.kernel.assign(np.zeros(model.eos_head.kernel.shape, dtype="float32"))
    model.eos_head.bias.assign(np.full(model.eos_head.bias.shape, logit, dtype="float32"))


@pytest.mark.parametrize("squeeze_crossentropy", [False, True])
def test_joint_loss_equals_negative_log_prob(eos_approximator, eos_data, monkeypatch, squeeze_crossentropy):
    if squeeze_crossentropy:
        binary_crossentropy = keras.ops.binary_crossentropy

        def mps_binary_crossentropy(targets, logits, from_logits=False):
            # Keras' Torch MPS backend squeezes singleton output dimensions.
            if len(logits.shape) > 1 and logits.shape[-1] == 1:
                targets = keras.ops.squeeze(targets, axis=-1)
                logits = keras.ops.squeeze(logits, axis=-1)
            return binary_crossentropy(targets, logits, from_logits=from_logits)

        monkeypatch.setattr(keras.ops, "binary_crossentropy", mps_binary_crossentropy)

    loss = keras.ops.convert_to_numpy(metrics(eos_approximator, eos_data)["loss"])
    np.testing.assert_allclose(loss, -np.mean(eos_approximator.log_prob(eos_data)), rtol=1e-5)


def test_only_first_eos_contributes(eos_approximator, eos_data):
    set_stop_logit(eos_approximator, 0.0)
    values, mask, _, _ = eos_approximator._prepare_sequence(eos_data["inference_variables"], eos_data["inference_mask"])
    conditions, _ = eos_approximator.condition_builder.resolve(
        standardizer=eos_approximator.standardizer,
        encoder_network=eos_approximator.encoder_network,
        decoder_network=eos_approximator.decoder_network,
        inference_variables=values,
        inference_conditions=keras.ops.convert_to_tensor(eos_data["inference_conditions"]),
        summary_variables=None,
        stage="inference",
        inference_mask=mask,
    )
    packet_log_prob = keras.ops.convert_to_numpy(
        eos_approximator.inference_network.log_prob(values, conditions=conditions)
    )
    lengths = eos_data["inference_mask"].sum(axis=1)
    expected = np.where(eos_data["inference_mask"], packet_log_prob, 0).sum(axis=1) - (lengths + 1) * np.log(2)
    np.testing.assert_allclose(eos_approximator.log_prob(eos_data), expected, rtol=1e-5)


@pytest.mark.parametrize("padding", [1e20, np.nan])
def test_padding_ignored_by_loss_and_density(eos_approximator, eos_data, padding):
    changed = {key: value.copy() for key, value in eos_data.items()}
    changed["inference_variables"][~changed["inference_mask"]] = padding
    np.testing.assert_allclose(eos_approximator.log_prob(changed), eos_approximator.log_prob(eos_data))
    np.testing.assert_allclose(
        keras.ops.convert_to_numpy(metrics(eos_approximator, changed)["loss"]),
        keras.ops.convert_to_numpy(metrics(eos_approximator, eos_data)["loss"]),
    )


def test_mask_inferred_from_reserved_packet_rows(eos_approximator, eos_data):
    implicit = {key: value for key, value in eos_data.items() if key != "inference_mask"}
    np.testing.assert_allclose(eos_approximator.log_prob(implicit), eos_approximator.log_prob(eos_data))
    np.testing.assert_allclose(
        keras.ops.convert_to_numpy(metrics(eos_approximator, implicit)["loss"]),
        keras.ops.convert_to_numpy(metrics(eos_approximator, eos_data)["loss"]),
    )


def test_explicit_mask_disambiguates_sentinel(eos_approximator, eos_data):
    eos_approximator.eos_value = 0.0
    data = {key: value.copy() for key, value in eos_data.items()}
    data["inference_variables"][1, 0] = 0.0
    assert np.all(np.isfinite(eos_approximator.log_prob(data)))


@pytest.mark.parametrize("marker", [[-999.0, 0.0], np.nan, np.inf])
def test_vector_and_nonfinite_markers(eos_approximator, eos_data, marker):
    eos_approximator.eos_value = marker
    data = {key: value.copy() for key, value in eos_data.items()}
    data["inference_variables"][~data["inference_mask"]] = marker
    implicit = {key: value for key, value in data.items() if key != "inference_mask"}
    np.testing.assert_allclose(eos_approximator.log_prob(implicit), eos_approximator.log_prob(data))
    set_stop_logit(eos_approximator, 100.0)
    sampled = eos_approximator.sample(
        num_samples=2, max_horizon=4, conditions={"inference_conditions": data["inference_conditions"]}
    )
    np.testing.assert_allclose(sampled["inference_variables"], np.broadcast_to(marker, (3, 2, 4, 2)))


@pytest.mark.parametrize("shape, cap, expected", [("infer", 4, 4), (4, None, 4), ((4,), 3, 3)])
def test_sampling_horizon_alternatives(eos_approximator, eos_data, shape, cap, expected):
    result = eos_approximator.sample(
        num_samples=2,
        sample_shape=shape,
        max_horizon=cap,
        conditions={"inference_conditions": eos_data["inference_conditions"]},
    )
    assert result["inference_variables"].shape == (3, 2, expected, 2)


@pytest.mark.parametrize("logit, expected_length, truncated", [(100.0, 0, False), (-100.0, 4, True)])
def test_all_stop_or_cap(eos_approximator, eos_data, logit, expected_length, truncated):
    set_stop_logit(eos_approximator, logit)
    result = eos_approximator.sample(
        num_samples=3,
        sample_shape=4,
        conditions={"inference_conditions": eos_data["inference_conditions"]},
        seed=12,
    )
    assert result["inference_variables"].shape == (3, 3, 4, 2)
    np.testing.assert_array_equal(result["_lengths"], np.full((3, 3), expected_length))
    np.testing.assert_array_equal(result["_truncated"], np.full((3, 3), truncated))
    if expected_length == 0:
        np.testing.assert_array_equal(result["inference_variables"], -999.0)


def test_draws_stop_independently_and_stay_stopped(eos_approximator, eos_data):
    set_stop_logit(eos_approximator, 0.0)
    kwargs = dict(
        num_samples=256,
        sample_shape=4,
        batch_size=1,
        seed=123,
        conditions={"inference_conditions": eos_data["inference_conditions"]},
    )
    result = eos_approximator.sample(**kwargs)
    repeated = eos_approximator.sample(**kwargs)
    for key in result:
        np.testing.assert_array_equal(result[key], repeated[key])
    assert len(np.unique(result["_lengths"])) >= 3
    expected_mask = np.arange(4)[None, None, :] < result["_lengths"][..., None]
    np.testing.assert_array_equal(result["_mask"], expected_mask)
    np.testing.assert_array_equal(result["_truncated"], result["_lengths"] == 4)
    np.testing.assert_array_equal(result["inference_variables"][~expected_mask], -999.0)


@pytest.mark.parametrize("bad_mask", [np.ones((3, 4), bool), np.tile([True, False, True, False], (3, 1))])
def test_invalid_or_unterminated_sequence_has_zero_probability(eos_approximator, eos_data, bad_mask):
    data = eos_data | {"inference_mask": bad_mask}
    assert np.all(np.isneginf(eos_approximator.log_prob(data)))
    assert np.isposinf(keras.ops.convert_to_numpy(metrics(eos_approximator, data)["loss"]))


def test_sampling_requires_cap_and_rejects_known_lengths(eos_approximator, eos_data):
    conditions = {"inference_conditions": eos_data["inference_conditions"]}
    with pytest.raises(ValueError, match="explicit maximum horizon"):
        eos_approximator.sample(num_samples=2, conditions=conditions)
    with pytest.raises(ValueError, match="positive"):
        eos_approximator.sample(num_samples=2, conditions=conditions, sample_shape=0)
    with pytest.raises(ValueError, match="positive"):
        eos_approximator.sample(num_samples=2, conditions=conditions, max_horizon=0)
    with pytest.raises(ValueError, match="lengths are generated"):
        eos_approximator.sample(num_samples=2, conditions=eos_data, sample_shape=4)


@pytest.mark.parametrize("sequence_conditions", [False, True])
def test_conditions_only_without_eos(sequence_conditions, eos_data):
    conditions = eos_data["inference_conditions"]
    if sequence_conditions:
        conditions = np.repeat(conditions[:, None], 2, axis=1)
    model = AutoregressiveApproximator(
        inference_network=CouplingFlow(depth=1, permutation=None, use_actnorm=False, subnet_kwargs={"widths": (8,)}),
        encoder_network=keras.layers.Identity(),
        decoder_network=TransformerDecoder(
            embed_dim=8, num_heads=2, num_layers=1, dropout=0.0, include_condition=False
        ),
        standardize=None,
    )
    data = eos_data | {"inference_conditions": conditions}
    model.build({key: value.shape for key, value in data.items()})
    assert np.isfinite(keras.ops.convert_to_numpy(metrics(model, data)["loss"]))
    assert model.sample(num_samples=2, conditions={"inference_conditions": conditions}, sample_shape=4)[
        "inference_variables"
    ].shape == (3, 2, 4, 2)
    inferred = 2 if sequence_conditions else 1
    assert model.sample(num_samples=2, conditions={"inference_conditions": conditions})[
        "inference_variables"
    ].shape == (3, 2, inferred, 2)


@pytest.mark.parametrize("eos_value", [None, -999.0, [-999.0, 0.0]])
def test_default_transformer_encoder_and_recurrent_decoder(eos_data, eos_value):
    model = AutoregressiveApproximator(
        inference_network=CouplingFlow(
            depth=1, permutation=None, use_actnorm=False, subnet_kwargs={"widths": (8,), "dropout": 0.0}
        ),
        standardize=None,
        eos_value=eos_value,
    )
    assert isinstance(model.encoder_network, TimeSeriesTransformer)
    assert model.encoder_network.return_sequences is True
    assert isinstance(model.decoder_network, RecurrentDecoder)
    assert model.decoder_network.include_condition is True
    assert model.get_config()["eos_value"] == eos_value

    data = {
        "inference_variables": eos_data["inference_variables"].copy(),
        "inference_conditions": np.repeat(eos_data["inference_conditions"][:, None], 4, axis=1),
    }
    data["inference_variables"][~eos_data["inference_mask"]] = 0.0 if eos_value is None else eos_value
    model.build_from_data(data)
    assert np.isfinite(keras.ops.convert_to_numpy(metrics(model, data)["loss"]))
    assert np.all(np.isfinite(model.log_prob(data)))
    kwargs = {} if eos_value is None else {"max_horizon": 4}
    sampled = model.sample(
        num_samples=3, conditions={"inference_conditions": data["inference_conditions"]}, seed=21, **kwargs
    )
    assert sampled["inference_variables"].shape == (3, 3, 4, 2)
    if eos_value is not None:
        expected_mask = np.arange(4)[None, None] < sampled["_lengths"][..., None]
        np.testing.assert_array_equal(sampled["_mask"], expected_mask)
        np.testing.assert_array_equal(sampled["_truncated"], sampled["_lengths"] == 4)
        np.testing.assert_array_equal(
            sampled["inference_variables"][~expected_mask],
            np.broadcast_to(eos_value, sampled["inference_variables"].shape)[~expected_mask],
        )
    restored = AutoregressiveApproximator.from_config(model.get_config())
    assert isinstance(restored.encoder_network, TimeSeriesTransformer)
    assert isinstance(restored.decoder_network, RecurrentDecoder)


def test_recurrent_decoder_rejects_unaligned_memory(eos_data):
    model = AutoregressiveApproximator(
        inference_network=CouplingFlow(depth=1),
        encoder_network=keras.layers.Identity(),
        decoder_network=RecurrentDecoder(embed_dim=8),
        eos_value=-999.0,
    )
    with pytest.raises(ValueError, match="time-aligned"):
        model.build({key: value.shape for key, value in eos_data.items()})


@pytest.mark.parametrize("concatenate", [False, True])
def test_jacobians_only_count_real_packets(eos_approximator, eos_data, concatenate):
    scale = np.float32(2.0)
    transformed = eos_data | {"inference_variables": eos_data["inference_variables"] * scale}
    expected = eos_approximator.log_prob(transformed)
    if concatenate:
        adapter = Adapter().as_time_series(["a", "b"]).concatenate(["a", "b"], into="inference_variables")
        data = {key: value for key, value in eos_data.items() if key != "inference_variables"}
        data.update(a=eos_data["inference_variables"][..., 0], b=eos_data["inference_variables"][..., 1])
    else:
        adapter, data = Adapter(), eos_data
    eos_approximator.adapter = adapter.scale("inference_variables", by=scale)
    eos_approximator.adapter(data)  # Initialize reversible concatenation metadata.
    actual = eos_approximator.log_prob(data)
    expected += eos_data["inference_mask"].sum(axis=1) * 2 * np.log(scale)
    np.testing.assert_allclose(actual, expected, rtol=1e-5)


def test_nonlinear_adapter_jacobians_ignore_padding(eos_approximator, eos_data):
    data = {key: value.copy() for key, value in eos_data.items()}
    data["inference_variables"] = np.exp(data["inference_variables"] * data["inference_mask"][..., None])
    transformed = data | {"inference_variables": np.log(data["inference_variables"])}
    expected = eos_approximator.log_prob(transformed)
    expected -= np.where(data["inference_mask"][..., None], transformed["inference_variables"], 0).sum(axis=(1, 2))
    eos_approximator.adapter = Adapter().log("inference_variables")
    np.testing.assert_allclose(eos_approximator.log_prob(data), expected, rtol=1e-5)


def test_sample_metadata_bypasses_inverse_adapter(eos_approximator, eos_data):
    set_stop_logit(eos_approximator, 100.0)
    eos_approximator.adapter = Adapter().scale("inference_variables", by=2.0)
    result = eos_approximator.sample(
        num_samples=2,
        sample_shape=4,
        split=True,
        to_numpy=False,
        conditions={"inference_conditions": eos_data["inference_conditions"]},
        seed=10,
    )
    assert result["_lengths"].shape == (3, 2)
    assert result["_mask"].shape == (3, 2, 4)
    assert result["_truncated"].dtype == keras.ops.convert_to_tensor(False).dtype
    for key, value in result.items():
        assert keras.ops.is_tensor(value)
        if not key.startswith("_"):
            np.testing.assert_array_equal(keras.ops.convert_to_numpy(value), -999.0 / 2.0)


def test_empty_training_batch_preserves_standardization(eos_approximator, eos_data):
    config = eos_approximator.get_config() | {"standardize": "inference_variables"}
    eos_approximator = AutoregressiveApproximator.from_config(config)
    eos_approximator.build({key: value.shape for key, value in eos_data.items()})
    data = eos_data | {
        "inference_mask": np.zeros((3, 4), bool),
        "inference_variables": np.full((3, 4, 2), -999.0, dtype="float32"),
    }
    result = eos_approximator.compute_metrics(**keras.tree.map_structure(keras.ops.convert_to_tensor, data))
    assert np.isfinite(keras.ops.convert_to_numpy(result["loss"]))
    layer = eos_approximator.standardizer.standardize_layers["inference_variables"]
    assert float(keras.ops.convert_to_numpy(layer.count[0])) == 0.0
    np.testing.assert_array_equal(keras.ops.convert_to_numpy(layer.moving_mean[0]), 0.0)


def test_compiled_training_updates_stop_head(eos_approximator, eos_data):
    from bayesflow.datasets import OfflineDataset

    before = keras.ops.convert_to_numpy(eos_approximator.eos_head.kernel).copy()
    eos_approximator.compile(optimizer=keras.optimizers.Adam(1e-3), jit_compile=keras.backend.backend() != "torch")
    dataset = OfflineDataset(eos_data, batch_size=3, adapter=eos_approximator.adapter, shuffle=False)
    result = eos_approximator.fit(dataset=dataset, epochs=1, verbose=0)
    assert np.isfinite(result.history["loss"][0])
    after = keras.ops.convert_to_numpy(eos_approximator.eos_head.kernel)
    assert not np.array_equal(before, after)


@pytest.mark.parametrize("marker", [-999.0, [-999.0, 0.0]])
def test_save_load_preserves_eos(eos_approximator, eos_data, tmp_path, marker):
    eos_approximator.eos_value = marker
    expected = eos_approximator.log_prob(eos_data)
    sample_kwargs = dict(
        num_samples=3, sample_shape=4, seed=23, conditions={"inference_conditions": eos_data["inference_conditions"]}
    )
    expected_samples = eos_approximator.sample(**sample_kwargs)
    path = tmp_path / "autoregressive_eos.keras"
    keras.saving.save_model(eos_approximator, path)
    restored = keras.saving.load_model(path)
    assert restored.eos_value == marker
    np.testing.assert_allclose(restored.log_prob(eos_data), expected, rtol=1e-5)
    actual_samples = restored.sample(**sample_kwargs)
    for key in expected_samples:
        np.testing.assert_allclose(actual_samples[key], expected_samples[key], rtol=1e-5)


def test_sample_weights_apply_to_both_losses(eos_approximator, eos_data):
    weights = np.array([0.5, 1.0, 2.0], dtype="float32")
    actual = keras.ops.convert_to_numpy(metrics(eos_approximator, eos_data, sample_weight=weights)["loss"])
    expected = np.mean(-eos_approximator.log_prob(eos_data) * weights)
    np.testing.assert_allclose(actual, expected, rtol=1e-5)


def test_extra_eos_padding_does_not_change_joint_loss(eos_approximator, eos_data):
    padded = eos_data | {
        "inference_variables": np.pad(
            eos_data["inference_variables"], ((0, 0), (0, 2), (0, 0)), constant_values=-999.0
        ),
        "inference_mask": np.pad(eos_data["inference_mask"], ((0, 0), (0, 2)), constant_values=False),
    }
    np.testing.assert_allclose(eos_approximator.log_prob(padded), eos_approximator.log_prob(eos_data), rtol=1e-5)
    np.testing.assert_allclose(
        keras.ops.convert_to_numpy(metrics(eos_approximator, padded)["loss"]),
        keras.ops.convert_to_numpy(metrics(eos_approximator, eos_data)["loss"]),
        rtol=1e-5,
    )


def test_standardization_excludes_eos_and_restores_sample_sentinel(eos_approximator, eos_data):
    model = AutoregressiveApproximator.from_config(
        eos_approximator.get_config() | {"standardize": "inference_variables"}
    )
    model.build_from_data(eos_data)
    model.compute_metrics(**keras.tree.map_structure(keras.ops.convert_to_tensor, eos_data), stage="training")
    layer = model.standardizer.standardize_layers["inference_variables"]
    valid = eos_data["inference_variables"][eos_data["inference_mask"]]
    np.testing.assert_allclose(keras.ops.convert_to_numpy(layer.moving_mean[0]), valid.mean(axis=0), rtol=1e-5)
    assert float(keras.ops.convert_to_numpy(layer.count[0])) == len(valid)
    set_stop_logit(model, 100.0)
    result = model.sample(
        num_samples=2, sample_shape=4, conditions={"inference_conditions": eos_data["inference_conditions"]}
    )
    np.testing.assert_array_equal(result["inference_variables"], -999.0)


@pytest.mark.jax
@pytest.mark.tensorflow
def test_jit_sampling_keeps_fixed_shapes_and_seeded_outputs(eos_approximator, eos_data):
    from bayesflow._backend import jit

    seed = keras.random.SeedGenerator(234)
    seed_state = keras.ops.convert_to_tensor([234, 0], dtype=seed.state.dtype)

    def sample(conditions):
        # Create variables outside TF graphs; thread random state explicitly in
        # JAX instead of mutating a generator captured by a compiled closure.
        with keras.StatelessScope(state_mapping=[(seed.state, seed_state)]):
            return eos_approximator.sample(
                num_samples=2,
                max_horizon=4,
                seed=seed,
                to_numpy=False,
                conditions={"inference_conditions": conditions},
            )

    compiled = jit(sample)
    conditions = keras.ops.convert_to_tensor(eos_data["inference_conditions"])
    result = compiled(conditions)
    repeated = compiled(conditions)
    assert result["inference_variables"].shape == (3, 2, 4, 2)
    for key in result:
        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(result[key]), keras.ops.convert_to_numpy(repeated[key]), rtol=1e-5
        )
