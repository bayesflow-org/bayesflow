import keras
import numpy as np
import pytest

from bayesflow.masking import InferenceMasks, MaskHandler, MaskName


def _tensor(values):
    return keras.ops.convert_to_tensor(np.asarray(values, dtype=np.float32))


def _numpy(value):
    return keras.ops.convert_to_numpy(value)


def test_collect_inference_masks_routes_names_and_uses_later_precedence():
    first = {"inference_attention_mask": _tensor([1.0, 0.0])}
    second = {"inference_attention_mask": _tensor([0.0, 1.0]), "inference_mask": None}

    collected = MaskHandler.collect_inference_masks(first, second)

    assert set(collected) == {"attention_mask"}
    np.testing.assert_array_equal(_numpy(collected["attention_mask"]), np.asarray([0.0, 1.0], dtype=np.float32))


def test_collect_summary_masks_routes_attention_and_padding_masks():
    attention = _tensor([1.0, 0.0])
    padding = _tensor([0.0, 1.0])

    collected = MaskHandler.collect_summary_masks(
        {
            "summary_attention_mask": attention,
            "summary_mask": padding,
        }
    )

    assert set(collected) == {"attention_mask", "mask"}
    np.testing.assert_array_equal(_numpy(collected["attention_mask"]), _numpy(attention))
    np.testing.assert_array_equal(_numpy(collected["mask"]), _numpy(padding))


def test_collect_semantic_masks_uses_later_precedence():
    discovered = {
        MaskName.FIXED_TARGET: _tensor([1.0, 0.0]),
        MaskName.FIXED_TARGET_VALUE: _tensor([2.0, 3.0]),
        MaskName.ATTENTION_MASK: _tensor([0.0, 1.0]),
        MaskName.MASK: _tensor([1.0, 1.0]),
    }
    explicit = {MaskName.FIXED_TARGET_VALUE: _tensor([4.0, 5.0])}

    collected = MaskHandler.collect_semantic_masks(discovered, explicit)

    np.testing.assert_array_equal(
        _numpy(collected[MaskName.FIXED_TARGET_VALUE]),
        np.asarray([4.0, 5.0], dtype=np.float32),
    )
    assert set(collected) == {MaskName.FIXED_TARGET, MaskName.FIXED_TARGET_VALUE}


def test_with_inference_masks_routes_masks_and_preserves_passthrough():
    adapted_attention = _tensor([1.0, 0.0])
    explicit_attention = _tensor([0.0, 1.0])
    padding = _tensor([1.0, 1.0])
    adapted_fixed = _tensor([1.0, 0.0])
    explicit_fixed = _tensor([0.0, 1.0])

    prepared = MaskHandler.with_inference_masks(
        {
            MaskName.INFERENCE_ATTENTION: explicit_attention,
            MaskName.FIXED_TARGET: explicit_fixed,
            "steps": 4,
        },
        {
            MaskName.INFERENCE_ATTENTION: adapted_attention,
            MaskName.INFERENCE: padding,
            MaskName.FIXED_TARGET: adapted_fixed,
        },
    )

    assert set(prepared) == {
        MaskName.ATTENTION_MASK,
        MaskName.MASK,
        MaskName.FIXED_TARGET,
        "steps",
    }
    np.testing.assert_array_equal(_numpy(prepared[MaskName.ATTENTION_MASK]), _numpy(explicit_attention))
    np.testing.assert_array_equal(_numpy(prepared[MaskName.MASK]), _numpy(padding))
    np.testing.assert_array_equal(_numpy(prepared[MaskName.FIXED_TARGET]), _numpy(explicit_fixed))
    assert prepared["steps"] == 4


def test_inference_masks_expose_only_semantic_kwargs(fixed_target_mask):
    attention_mask = keras.ops.ones((2, 3, 3), dtype="float32")
    masks = InferenceMasks(
        fixed_target_mask=fixed_target_mask,
        subnet_kwargs={MaskName.ATTENTION_MASK: attention_mask},
    )

    assert set(masks.semantic_kwargs()) == {MaskName.FIXED_TARGET}


def test_collect_mask_kwargs_includes_canonical_subnet_and_semantic_fields():
    source = {
        MaskName.ATTENTION_MASK: _tensor([[1.0, 0.0]]),
        MaskName.MASK: _tensor([1.0, 0.0]),
        MaskName.FIXED_TARGET: _tensor([0.0, 1.0]),
        MaskName.INFER_TARGET: _tensor([1.0, 1.0]),
        MaskName.OBSERVED_CONDITION: _tensor([1.0, 0.0]),
        MaskName.FIXED_TARGET_VALUE: _tensor([2.0, 3.0]),
        "steps": 4,
    }

    assert set(MaskHandler.collect_mask_kwargs(source)) == {
        MaskName.ATTENTION_MASK,
        MaskName.MASK,
        MaskName.FIXED_TARGET,
        MaskName.INFER_TARGET,
        MaskName.OBSERVED_CONDITION,
        MaskName.FIXED_TARGET_VALUE,
    }


def test_remove_all_mask_kwargs():
    source = {
        "attention_mask": _tensor([[1.0, 0.0]]),
        "mask": _tensor([1.0, 0.0]),
        MaskName.FIXED_TARGET: _tensor([0.0, 1.0]),
        MaskName.FIXED_TARGET_VALUE: _tensor([2.0, 3.0]),
        "steps": 4,
    }

    assert MaskHandler.without_mask_kwargs(source) == {"steps": 4}


def test_mask_names_define_canonical_minimum_batch_ranks():
    expected = {
        MaskName.ATTENTION_MASK: 3,
        MaskName.MASK: 2,
        MaskName.FIXED_TARGET: 2,
        MaskName.INFER_TARGET: 2,
        MaskName.OBSERVED_CONDITION: 2,
        MaskName.FIXED_TARGET_VALUE: 2,
    }

    assert {name: MaskName.minimum_batch_rank(name) for name in expected} == expected
    assert MaskName.minimum_batch_rank(MaskName.INFERENCE_ATTENTION) is None


@pytest.mark.parametrize(
    ("name", "shape", "target_structural_rank", "expected"),
    [
        (MaskName.ATTENTION_MASK, (2, 4), 0, False),
        (MaskName.ATTENTION_MASK, (2, 3, 4), 0, True),
        (MaskName.MASK, (2,), 0, False),
        (MaskName.MASK, (2, 4), 0, True),
        (MaskName.FIXED_TARGET, (2, 3), 0, True),
        (MaskName.FIXED_TARGET, (2, 3), 1, False),
        (MaskName.FIXED_TARGET, (2, 4, 3), 1, True),
    ],
)
def test_has_batch_axis_uses_canonical_rank(name, shape, target_structural_rank, expected):
    value = keras.ops.ones(shape, dtype="float32")

    assert (
        MaskHandler.has_batch_axis(
            name,
            value,
            batch_size=2,
            target_structural_rank=target_structural_rank,
        )
        is expected
    )
    assert not MaskHandler.has_batch_axis(
        name,
        value,
        batch_size=None,
        target_structural_rank=target_structural_rank,
    )


def test_batched_keys_collects_only_batched_values():
    source = {
        MaskName.MASK: keras.ops.ones((2, 4), dtype="float32"),
        MaskName.ATTENTION_MASK: keras.ops.ones((2, 4), dtype="float32"),
        "steps": 4,
    }

    assert MaskHandler.batched_keys(source, batch_size=2) == {MaskName.MASK}


def test_batched_keys_checks_unknown_tensor_shapes():
    source = {
        "batched": keras.ops.ones((2, 3), dtype="float32"),
        "shared": keras.ops.ones((3,), dtype="float32"),
        "scalar": keras.ops.convert_to_tensor(np.float32(1.0)),
    }

    assert MaskHandler.batched_keys(source, batch_size=2) == {"batched"}
    assert MaskHandler.batched_keys(source, batch_size=None) == set()


def test_observed_condition_masks_use_condition_structure():
    vector_mask = keras.ops.ones((2, 3), dtype="float32")
    structured_mask = keras.ops.ones((2, 4, 3), dtype="float32")

    assert MaskHandler.has_batch_axis(
        MaskName.OBSERVED_CONDITION,
        vector_mask,
        batch_size=2,
        condition_structural_rank=0,
    )
    assert not MaskHandler.has_batch_axis(
        MaskName.OBSERVED_CONDITION,
        vector_mask,
        batch_size=2,
        condition_structural_rank=1,
    )
    assert MaskHandler.has_batch_axis(
        MaskName.OBSERVED_CONDITION,
        structured_mask,
        batch_size=2,
        condition_structural_rank=1,
    )


def test_infer_batch_size_uses_only_structurally_valid_target_fields():
    attention = keras.ops.ones((2, 3, 4), dtype="float32")
    flat_target = keras.ops.ones((2, 3), dtype="float32")
    structured_target = keras.ops.ones((2, 4, 3), dtype="float32")

    assert MaskHandler.infer_batch_size({MaskName.ATTENTION_MASK: attention}) is None
    assert MaskHandler.infer_batch_size({MaskName.FIXED_TARGET: flat_target}) == 2
    assert MaskHandler.infer_batch_size({MaskName.FIXED_TARGET: flat_target}, target_structural_rank=1) is None
    assert (
        MaskHandler.infer_batch_size(
            {MaskName.FIXED_TARGET_VALUE: structured_target},
            target_structural_rank=1,
        )
        == 2
    )


@pytest.mark.parametrize(
    ("name", "shape", "has_batch_axis", "step", "time_axis"),
    [
        (MaskName.ATTENTION_MASK, (2, 4), False, 1, None),
        (MaskName.ATTENTION_MASK, (2, 3, 4), True, 1, None),
        (MaskName.MASK, (2, 3, 4), True, 1, 1),
        (MaskName.ATTENTION_MASK, (3, 2, 4), False, 2, 0),
    ],
)
def test_select_autoregressive_step_preserves_event_masks_and_slices_temporal_masks(
    name,
    shape,
    has_batch_axis,
    step,
    time_axis,
):
    value = keras.ops.reshape(keras.ops.arange(np.prod(shape), dtype="float32"), shape)

    result = MaskHandler.select_autoregressive_step(
        name,
        value,
        step=step,
        num_steps=3,
        has_batch_axis=has_batch_axis,
    )

    if time_axis is None:
        assert result is value
    else:
        np.testing.assert_array_equal(_numpy(result), _numpy(keras.ops.take(value, step, axis=time_axis)))


@pytest.mark.parametrize(
    ("shape", "has_batch_axis", "expected_axis"),
    [
        ((3, 4), False, 0),
        ((2, 3, 4), True, 1),
        ((1, 4), False, 0),
        ((2, 1, 4), True, 1),
        ((4,), False, None),
        ((2, 4), True, None),
    ],
)
def test_select_autoregressive_step_handles_custom_values(shape, has_batch_axis, expected_axis):
    value = keras.ops.reshape(keras.ops.arange(np.prod(shape), dtype="float32"), shape)

    result = MaskHandler.select_autoregressive_step(
        "custom",
        value,
        step=2,
        num_steps=3,
        has_batch_axis=has_batch_axis,
    )

    if expected_axis is None:
        assert result is value
    else:
        index = 0 if shape[expected_axis] == 1 else 2
        np.testing.assert_array_equal(_numpy(result), _numpy(keras.ops.take(value, index, axis=expected_axis)))


def test_autoregressive_padding_mask_prefers_matching_batch_axis():
    value = keras.ops.ones((3, 4), dtype="float32")
    batched_keys = MaskHandler.batched_keys({MaskName.MASK: value}, batch_size=3)

    result = MaskHandler.select_autoregressive_step(
        MaskName.MASK,
        value,
        step=1,
        num_steps=3,
        has_batch_axis=MaskName.MASK in batched_keys,
    )

    assert batched_keys == {MaskName.MASK}
    assert result is value


@pytest.mark.parametrize(
    ("name", "shape", "expected_shape"),
    [
        (MaskName.ATTENTION_MASK, (2, 4), (2, 4)),
        (MaskName.ATTENTION_MASK, (2, 3, 4), (6, 3, 4)),
        (MaskName.ATTENTION_MASK, (1, 3, 4), (1, 3, 4)),
        (MaskName.MASK, (2,), (2,)),
        (MaskName.MASK, (2, 4), (6, 4)),
        (MaskName.MASK, (1, 4), (1, 4)),
    ],
)
def test_repeat_compositional_guesses_shared_and_batched_ordinary_masks(name, shape, expected_shape):
    value = keras.ops.reshape(keras.ops.arange(np.prod(shape), dtype="float32"), shape)

    repeated = MaskHandler().repeat_compositional(
        InferenceMasks(subnet_kwargs={name: value}),
        num_repeats=3,
    )

    result = repeated.subnet_kwargs[name]
    assert tuple(result.shape) == expected_shape
    if expected_shape == shape:
        np.testing.assert_array_equal(_numpy(result), _numpy(value))
    else:
        np.testing.assert_array_equal(_numpy(result), np.repeat(_numpy(value), 3, axis=0))


def test_repeat_compositional_repeats_semantic_masks_and_defers_conditions():
    fixed = _tensor([[1.0, 0.0], [0.0, 1.0]])
    inferred = _tensor([[1.0, 1.0], [0.0, 1.0]])
    fixed_value = _tensor([[2.0, 3.0], [4.0, 5.0]])
    observed = keras.ops.reshape(keras.ops.arange(12, dtype="float32"), (2, 3, 2))
    masks = InferenceMasks(
        fixed_target_mask=fixed,
        infer_target_mask=inferred,
        observed_condition_mask=observed,
        fixed_target_value=fixed_value,
        subnet_kwargs={
            MaskName.FIXED_TARGET: fixed,
            MaskName.INFER_TARGET: inferred,
            MaskName.OBSERVED_CONDITION: observed,
        },
    )

    repeated = MaskHandler().repeat_compositional(masks, num_repeats=3)

    np.testing.assert_array_equal(_numpy(repeated.fixed_target_mask), np.repeat(_numpy(fixed), 3, axis=0))
    np.testing.assert_array_equal(_numpy(repeated.infer_target_mask), np.repeat(_numpy(inferred), 3, axis=0))
    np.testing.assert_array_equal(_numpy(repeated.fixed_target_value), np.repeat(_numpy(fixed_value), 3, axis=0))
    assert repeated.observed_condition_mask is None
    np.testing.assert_array_equal(
        _numpy(repeated.subnet_kwargs[MaskName.FIXED_TARGET]),
        np.repeat(_numpy(fixed), 3, axis=0),
    )
    np.testing.assert_array_equal(
        _numpy(repeated.subnet_kwargs[MaskName.INFER_TARGET]),
        np.repeat(_numpy(inferred), 3, axis=0),
    )
    assert repeated.subnet_kwargs[MaskName.OBSERVED_CONDITION] is None
    assert MaskName.FIXED_TARGET_VALUE not in repeated.subnet_kwargs

    prepared = MaskHandler.with_flattened_condition_mask(repeated, observed)
    expected_observed = _numpy(observed).reshape(6, 2)
    np.testing.assert_array_equal(_numpy(prepared.observed_condition_mask), expected_observed)
    np.testing.assert_array_equal(_numpy(prepared.subnet_kwargs[MaskName.OBSERVED_CONDITION]), expected_observed)


@pytest.mark.parametrize("stage", ["validation", "inference"])
def test_resolve_does_not_generate_masks_outside_training(targets, conditions, stage):
    handler = MaskHandler(
        fixed_target_prob=1.0,
        missing_target_prob=1.0,
        missing_conditions_prob=1.0,
        seed_generator=keras.random.SeedGenerator(12),
    )
    state_before = _numpy(handler.seed_generator.state).copy()

    masks = handler.resolve(targets=targets, conditions=conditions, stage=stage)

    assert masks == InferenceMasks()
    np.testing.assert_array_equal(_numpy(handler.seed_generator.state), state_before)


def test_resolve_generates_consistent_float32_masks(targets, conditions):
    handler = MaskHandler(
        fixed_target_prob=1.0,
        missing_target_prob=1.0,
        missing_conditions_prob=1.0,
        seed_generator=keras.random.SeedGenerator(21),
    )

    masks = handler.resolve(targets=targets, conditions=conditions, stage="training")
    fixed = _numpy(masks.fixed_target_mask)
    inferred = _numpy(masks.infer_target_mask)
    observed = _numpy(masks.observed_condition_mask)

    assert fixed.dtype == np.float32
    assert inferred.dtype == np.float32
    assert observed.dtype == np.float32
    assert np.all(fixed.sum(axis=-1) >= 1.0)
    assert np.all(inferred >= fixed)
    np.testing.assert_array_equal(observed, np.zeros_like(observed, dtype=np.float32))


def test_random_mask_is_a_public_handler_operation(targets):
    mask = MaskHandler(seed_generator=keras.random.SeedGenerator(22)).random_mask(
        targets,
        drop_prob=1.0,
        keep_one=True,
    )

    assert _numpy(mask).dtype == np.float32
    assert np.all(_numpy(mask).sum(axis=-1) >= 1.0)


def test_explicit_masks_override_generation(targets, conditions, fixed_target_mask):
    handler = MaskHandler(
        fixed_target_prob=1.0,
        missing_target_prob=1.0,
        missing_conditions_prob=1.0,
        seed_generator=keras.random.SeedGenerator(7),
    )
    infer = keras.ops.ones_like(targets)
    observed = keras.ops.ones_like(conditions)
    masks = handler.resolve(
        {
            MaskName.FIXED_TARGET: fixed_target_mask,
            MaskName.INFER_TARGET: infer,
            MaskName.OBSERVED_CONDITION: observed,
        },
        targets=targets,
        conditions=conditions,
        stage="training",
    )

    np.testing.assert_array_equal(_numpy(masks.fixed_target_mask), _numpy(fixed_target_mask))
    np.testing.assert_array_equal(_numpy(masks.infer_target_mask), _numpy(infer))
    np.testing.assert_array_equal(_numpy(masks.observed_condition_mask), _numpy(observed))


def test_explicit_infer_target_mask_constrains_generated_fixed_targets(targets):
    infer = _tensor([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
    handler = MaskHandler(
        fixed_target_prob=1.0,
        seed_generator=keras.random.SeedGenerator(31),
    )

    masks = handler.resolve(
        {MaskName.INFER_TARGET: infer},
        targets=targets,
        stage="training",
    )

    np.testing.assert_array_equal(_numpy(masks.infer_target_mask), _numpy(infer))
    np.testing.assert_array_equal(_numpy(masks.fixed_target_mask), _numpy(infer))


def test_explicit_fixed_targets_mix_with_generated_missingness(targets, conditions, fixed_target_mask):
    class MaskedSubnet(keras.Layer):
        def call(self, inputs, infer_target_mask=None, observed_condition_mask=None):
            return inputs[0]

    handler = MaskHandler(
        missing_target_prob=1.0,
        missing_conditions_prob=1.0,
        seed_generator=keras.random.SeedGenerator(32),
    )

    masks = handler.resolve(
        {MaskName.FIXED_TARGET: fixed_target_mask},
        targets=targets,
        conditions=conditions,
        stage="training",
        subnet=MaskedSubnet(),
    )

    np.testing.assert_array_equal(_numpy(masks.fixed_target_mask), _numpy(fixed_target_mask))
    np.testing.assert_array_equal(_numpy(masks.infer_target_mask), _numpy(fixed_target_mask))
    np.testing.assert_array_equal(_numpy(masks.observed_condition_mask), np.zeros_like(_numpy(conditions)))


def test_local_seed_is_reproducible_and_advances_between_calls(targets):
    first = MaskHandler(seed_generator=keras.random.SeedGenerator(33))
    second = MaskHandler(seed_generator=keras.random.SeedGenerator(33))
    initial_state = _numpy(first.seed_generator.state).copy()

    first_draw = first.random_mask(targets, drop_prob=0.5)
    first_state = _numpy(first.seed_generator.state).copy()
    second_draw = first.random_mask(targets, drop_prob=0.5)
    second_state = _numpy(first.seed_generator.state).copy()

    np.testing.assert_array_equal(_numpy(first_draw), _numpy(second.random_mask(targets, drop_prob=0.5)))
    np.testing.assert_array_equal(_numpy(second_draw), _numpy(second.random_mask(targets, drop_prob=0.5)))
    assert not np.array_equal(first_state, initial_state)
    assert not np.array_equal(second_state, first_state)


def test_missing_conditions_without_conditions_does_not_advance_random_state(targets):
    seed_generator = keras.random.SeedGenerator(34)
    handler = MaskHandler(
        missing_conditions_prob=1.0,
        seed_generator=seed_generator,
    )
    state_before = _numpy(seed_generator.state).copy()

    masks = handler.resolve(targets=targets, conditions=None, stage="training")

    assert masks.observed_condition_mask is None
    np.testing.assert_array_equal(_numpy(seed_generator.state), state_before)


def test_maybe_apply_masks_and_replaces_values(targets, fixed_target_mask):
    handler = MaskHandler()
    fixed_value = keras.ops.full_like(targets, keras.ops.cast(9.0, targets.dtype))

    replaced = handler.maybe_apply(targets, fixed_target_mask, replacement=fixed_value)
    projected = handler.maybe_apply(keras.ops.ones_like(targets), fixed_target_mask)

    expected = _numpy(targets).copy()
    expected[:, 1] = np.float32(9.0)
    np.testing.assert_array_equal(_numpy(replaced), expected)
    np.testing.assert_array_equal(_numpy(projected), _numpy(fixed_target_mask))
    assert handler.maybe_apply(targets, None) is targets


def test_maybe_apply_isolates_nan_placeholders():
    handler = MaskHandler()
    values = _tensor([[2.0, np.nan, 4.0]])
    replacement = _tensor([[np.nan, 7.0, np.nan]])
    mask = _tensor([[1.0, 0.0, 1.0]])

    replaced = _numpy(handler.maybe_apply(values, mask, replacement))
    projected = _numpy(handler.maybe_apply(values, mask))

    np.testing.assert_array_equal(replaced, np.asarray([[2.0, 7.0, 4.0]], dtype=np.float32))
    np.testing.assert_array_equal(projected, np.asarray([[2.0, 0.0, 4.0]], dtype=np.float32))
    assert replaced.dtype == np.float32
    assert projected.dtype == np.float32


def test_subnet_capability_requires_explicit_parameters(fixed_target_mask):
    class ExplicitSubnet(keras.Layer):
        def call(self, inputs, fixed_target_mask=None, attention_mask=None, **kwargs):
            return inputs[0]

    class CatchAllSubnet(keras.Layer):
        def call(self, inputs, **kwargs):
            return inputs[0]

    attention = keras.ops.ones((2, 3, 3), dtype="float32")
    source = {
        MaskName.FIXED_TARGET: fixed_target_mask,
        "attention_mask": attention,
    }

    assert set(MaskHandler.collect_subnet_masks(ExplicitSubnet(), source)) == {
        "attention_mask",
        MaskName.FIXED_TARGET,
    }
    assert MaskHandler.collect_subnet_masks(CatchAllSubnet(), source) == {}


def test_resolve_routes_only_explicit_subnet_masks(targets, fixed_target_mask):
    class ExplicitSubnet(keras.Layer):
        def call(self, inputs, fixed_target_mask=None):
            return inputs[0]

    source = {
        MaskName.FIXED_TARGET: fixed_target_mask,
        MaskName.INFER_TARGET: keras.ops.ones_like(targets),
    }
    masks = MaskHandler().resolve(source, targets=targets, subnet=ExplicitSubnet())

    assert set(masks.subnet_kwargs) == {MaskName.FIXED_TARGET}


@pytest.mark.parametrize("name", ["attention_mask", "mask"])
def test_resolve_raises_if_subnet_does_not_name_the_mask(targets, name):
    class OnlyKwargs(keras.Layer):
        def call(self, inputs, **kwargs):
            return inputs[0]

    with pytest.raises(ValueError, match=f"`{name}` would be ignored"):
        MaskHandler().resolve({name: keras.ops.ones((2, 3))}, targets=targets, subnet=OnlyKwargs())


def test_resolve_without_subnet_does_not_check_masks(targets):
    # e.g. the approximator resolves masks before a subnet is involved
    MaskHandler().resolve({"mask": keras.ops.ones((2, 3), dtype="float32")}, targets=targets)


def test_resolve_does_not_sample_masks_unsupported_by_subnet(targets, conditions):
    class CatchAllSubnet(keras.Layer):
        def call(self, inputs, **kwargs):
            return inputs[0]

    seed_generator = keras.random.SeedGenerator(23)
    handler = MaskHandler(
        missing_conditions_prob=1.0,
        seed_generator=seed_generator,
    )
    state_before = _numpy(seed_generator.state).copy()

    masks = handler.resolve(
        targets=targets,
        conditions=conditions,
        stage="training",
        subnet=CatchAllSubnet(),
    )

    assert masks.observed_condition_mask is None
    np.testing.assert_array_equal(_numpy(seed_generator.state), state_before)


def test_resolve_generates_missingness_only_when_subnet_supports_it(
    targets,
    conditions,
    fixed_target_mask,
):
    class UnmaskedSubnet(keras.Layer):
        def call(self, inputs):
            return inputs[0]

    seed_generator = keras.random.SeedGenerator(24)
    handler = MaskHandler(
        missing_target_prob=1.0,
        missing_conditions_prob=1.0,
        seed_generator=seed_generator,
    )
    state_before = _numpy(seed_generator.state).copy()

    masks = handler.resolve(
        {MaskName.FIXED_TARGET: fixed_target_mask},
        targets=targets,
        conditions=conditions,
        stage="training",
        subnet=UnmaskedSubnet(),
    )

    assert masks.infer_target_mask is None
    assert masks.observed_condition_mask is None
    np.testing.assert_array_equal(_numpy(seed_generator.state), state_before)


def test_with_flattened_condition_mask_flattens_and_routes_mask():
    observed = keras.ops.reshape(keras.ops.arange(12, dtype="float32"), (2, 3, 2))
    masks = InferenceMasks(subnet_kwargs={MaskName.OBSERVED_CONDITION: observed})

    updated = MaskHandler.with_flattened_condition_mask(masks, observed)
    expected = _numpy(observed).reshape(6, 2)

    np.testing.assert_array_equal(_numpy(updated.observed_condition_mask), expected)
    np.testing.assert_array_equal(_numpy(updated.subnet_kwargs[MaskName.OBSERVED_CONDITION]), expected)


def test_integrator_callbacks_project_state_and_updates(targets, fixed_target_mask):
    fixed_value = keras.ops.full_like(targets, keras.ops.cast(9.0, targets.dtype))
    masks = InferenceMasks(fixed_target_mask=fixed_target_mask, fixed_target_value=fixed_value)
    handler = MaskHandler()

    project_state, project_updates = handler.integrator_callbacks(masks)
    state = {"xz": targets, "other": keras.ops.ones((2,), dtype="float32")}
    projected_state = project_state(state)
    projected_updates = project_updates(state)

    expected_state = _numpy(targets).copy()
    expected_state[:, 1] = np.float32(9.0)
    expected_updates = _numpy(targets).copy()
    expected_updates[:, 1] = np.float32(0.0)
    np.testing.assert_array_equal(_numpy(projected_state["xz"]), expected_state)
    np.testing.assert_array_equal(_numpy(projected_updates["xz"]), expected_updates)
    assert projected_state["other"] is state["other"]
    assert handler.integrator_callbacks(InferenceMasks()) == (None, None)
