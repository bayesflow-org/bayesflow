import keras
import numpy as np
import pytest

from bayesflow.approximators.helpers import Sampler
from bayesflow.masking import MaskHandler, MaskName


def _numpy(value):
    return keras.ops.convert_to_numpy(value)


def _values(shape):
    return np.arange(np.prod(shape), dtype=np.float32).reshape(shape)


class RecordingNetwork:
    def __init__(self):
        self.calls = []

    def sample(self, batch_shape, conditions=None, seed=None, **kwargs):
        self.calls.append(
            {
                "batch_shape": tuple(int(size) for size in batch_shape),
                "conditions": None if conditions is None else _numpy(conditions).copy(),
                "kwargs": {name: _numpy(value).copy() for name, value in kwargs.items()},
            }
        )
        return keras.ops.zeros((*batch_shape, 1), dtype="float32")


@pytest.mark.parametrize(
    ("name", "event_shape"),
    [
        (MaskName.MASK, (4,)),
        (MaskName.ATTENTION_MASK, (3, 4)),
        (MaskName.FIXED_TARGET, (4,)),
        (MaskName.FIXED_TARGET_VALUE, (4,)),
        (MaskName.INFER_TARGET, (4,)),
        (MaskName.OBSERVED_CONDITION, (4,)),
    ],
)
def test_sampler_slices_and_repeats_batched_masks(name, event_shape):
    network = RecordingNetwork()
    values = _values((3, *event_shape))

    samples = Sampler().sample(
        network,
        num_samples=2,
        conditions=_values((3, 2)),
        batch_size=2,
        sample_shape=(),
        seed=keras.random.SeedGenerator(101),
        **{name: values},
    )

    assert samples.shape == (3, 2, 1)
    assert _numpy(samples).dtype == np.float32
    assert [call["batch_shape"] for call in network.calls] == [(4,), (2,)]
    np.testing.assert_array_equal(
        network.calls[0]["kwargs"][name],
        np.repeat(values[:2], repeats=2, axis=0),
    )
    np.testing.assert_array_equal(
        network.calls[1]["kwargs"][name],
        np.repeat(values[2:], repeats=2, axis=0),
    )


@pytest.mark.parametrize(
    ("name", "event_shape"),
    [
        (MaskName.MASK, (4,)),
        (MaskName.ATTENTION_MASK, (3, 4)),
        (MaskName.FIXED_TARGET, (4,)),
        (MaskName.FIXED_TARGET_VALUE, (4,)),
        (MaskName.INFER_TARGET, (4,)),
        (MaskName.OBSERVED_CONDITION, (4,)),
    ],
)
@pytest.mark.parametrize("singleton", [False, True], ids=["shared", "singleton"])
def test_sampler_preserves_shared_and_singleton_masks(name, event_shape, singleton):
    network = RecordingNetwork()
    shape = (1, *event_shape) if singleton else event_shape
    values = _values(shape)

    Sampler().sample(
        network,
        num_samples=2,
        conditions=_values((3, 2)),
        batch_size=2,
        sample_shape=(),
        seed=keras.random.SeedGenerator(102),
        **{name: values},
    )

    assert len(network.calls) == 2
    for call in network.calls:
        np.testing.assert_array_equal(call["kwargs"][name], values)


@pytest.mark.parametrize(
    ("name", "event_shape"),
    [
        (MaskName.MASK, (3,)),
        (MaskName.ATTENTION_MASK, (3, 3)),
        (MaskName.FIXED_TARGET, (3, 2)),
        (MaskName.FIXED_TARGET_VALUE, (3, 2)),
        (MaskName.INFER_TARGET, (3, 2)),
        (MaskName.OBSERVED_CONDITION, (3, 2)),
    ],
)
def test_sampler_does_not_confuse_shared_structure_with_equal_batch_size(name, event_shape):
    network = RecordingNetwork()
    values = _values(event_shape)

    samples = Sampler().sample(
        network,
        num_samples=2,
        conditions=_values((3, 3, 2)),
        batch_size=2,
        sample_shape=(3,),
        seed=keras.random.SeedGenerator(103),
        **{name: values},
    )

    assert samples.shape == (3, 2, 3, 1)
    for call in network.calls:
        np.testing.assert_array_equal(call["kwargs"][name], values)


def test_sampler_batches_vector_condition_masks_with_structured_targets():
    network = RecordingNetwork()
    observed = _values((3, 2))

    samples = Sampler().sample(
        network,
        num_samples=2,
        conditions=_values((3, 2)),
        batch_size=2,
        sample_shape=(4,),
        seed=keras.random.SeedGenerator(105),
        observed_condition_mask=observed,
    )

    assert samples.shape == (3, 2, 4, 1)
    np.testing.assert_array_equal(
        network.calls[0]["kwargs"][MaskName.OBSERVED_CONDITION],
        np.repeat(observed[:2], repeats=2, axis=0),
    )
    np.testing.assert_array_equal(
        network.calls[1]["kwargs"][MaskName.OBSERVED_CONDITION],
        np.repeat(observed[2:], repeats=2, axis=0),
    )


@pytest.mark.parametrize("sample_shape", [4, (2, 3)], ids=["scalar", "structured"])
@pytest.mark.parametrize(
    "name",
    [MaskName.FIXED_TARGET, MaskName.FIXED_TARGET_VALUE, MaskName.INFER_TARGET],
)
def test_sampler_batches_unconditional_data_inferred_from_semantic_masks(sample_shape, name):
    network = RecordingNetwork()
    structural_shape = (sample_shape,) if isinstance(sample_shape, int) else sample_shape
    structural_rank = len(structural_shape)
    values = _values((5, *structural_shape, 2))
    source = {name: values}
    data_batch_size = MaskHandler.infer_batch_size(source, structural_rank)

    samples = Sampler().sample(
        network,
        num_samples=2,
        conditions=None,
        batch_size=2,
        data_batch_size=data_batch_size,
        sample_shape=sample_shape,
        seed=keras.random.SeedGenerator(104),
        **source,
    )

    assert data_batch_size == 5
    assert samples.shape == (5, 2, *structural_shape, 1)
    assert _numpy(samples).dtype == np.float32
    assert [call["batch_shape"] for call in network.calls] == [
        (4, *structural_shape),
        (4, *structural_shape),
        (2, *structural_shape),
    ]
    for call, expected in zip(network.calls, (values[:2], values[2:4], values[4:]), strict=True):
        assert call["conditions"] is None
        np.testing.assert_array_equal(
            call["kwargs"][name],
            np.repeat(expected, repeats=2, axis=0),
        )
