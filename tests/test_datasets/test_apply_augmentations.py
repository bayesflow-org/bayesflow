import pytest
from bayesflow.datasets._augmentations import apply_augmentations


def test_none_returns_same_object():
    batch = {"x": [1, 2], "y": [3, 4]}
    out = apply_augmentations(batch, None)
    assert out is batch
    assert out == {"x": [1, 2], "y": [3, 4]}


def test_mapping_iterates_all_items_and_preserves_other_keys():
    batch = {"x": 1, "y": 2, "z": 999}
    augs = {"x": lambda v: v * 10, "y": lambda v: v - 1}
    out = apply_augmentations(batch, augs)

    assert out is batch
    assert out["x"] == 10
    assert out["y"] == 1
    assert out["z"] == 999  # untouched


def test_mapping_calls_each_fn_with_value_for_that_key():
    batch = {"x": "abc", "y": "def"}

    seen = {}

    def fx(v):
        seen["x_arg"] = v
        return v.upper()

    def fy(v):
        seen["y_arg"] = v
        return v[::-1]

    out = apply_augmentations(batch, {"x": fx, "y": fy})

    assert out is batch
    assert seen == {"x_arg": "abc", "y_arg": "def"}
    assert out == {"x": "ABC", "y": "fed"}


def test_sequence_applies_in_order():
    batch = {"x": 1}

    def a1(b):
        return {**b, "x": b["x"] + 1}

    def a2(b):
        return {**b, "x": b["x"] * 10}

    out = apply_augmentations(batch, [a1, a2])

    assert out == {"x": 20}


def test_sequence_accepts_tuples_and_other_sequences():
    batch = {"x": 3}

    def a1(b):
        return {**b, "x": b["x"] - 1}

    def a2(b):
        return {**b, "x": b["x"] * 2}

    out = apply_augmentations(batch, (a1, a2))
    assert out == {"x": 4}


@pytest.mark.parametrize("bad", ["abc", b"bytes"])
def test_string_or_bytes_are_not_treated_as_sequence_and_raise(bad):
    batch = {"x": 1}
    with pytest.raises(RuntimeError, match=r"Could not apply augmentations of type"):
        apply_augmentations(batch, bad)


def test_callable_single_fn_applies_to_whole_batch():
    batch = {"x": 1, "y": 2}

    def aug(b):
        return {"sum": b["x"] + b["y"]}

    out = apply_augmentations(batch, aug)

    assert out == {"sum": 3}
    assert out is not batch


def test_unknown_type_raises_runtimeerror_with_type_in_message():
    batch = {"x": 1}
    augmentations = 123  # not None/Mapping/Sequence[Callable]/Callable
    with pytest.raises(RuntimeError):
        apply_augmentations(batch, augmentations)


def test_mapping_raises_keyerror_if_batch_missing_key():
    batch = {"x": 1}
    augs = {"missing": lambda v: v}
    with pytest.raises(KeyError):
        apply_augmentations(batch, augs)
