from enum import StrEnum


class MaskName(StrEnum):
    """Canonical mask keyword names.

    Semantic masks use ``1`` for an active coordinate and ``0`` for a masked
    one. Thus, ``fixed_target_mask`` is ``0`` where a target is fixed,
    ``infer_target_mask`` is ``1`` where a target is present, and
    ``observed_condition_mask`` is ``1`` where a condition is observed.
    """

    SUMMARY_ATTENTION = "summary_attention_mask"
    SUMMARY = "summary_mask"
    INFERENCE_ATTENTION = "inference_attention_mask"
    INFERENCE = "inference_mask"
    INFER_TARGET = "infer_target_mask"
    FIXED_TARGET = "fixed_target_mask"
    OBSERVED_CONDITION = "observed_condition_mask"
    FIXED_TARGET_VALUE = "fixed_target_value"
    ATTENTION_MASK = "attention_mask"
    MASK = "mask"

    @classmethod
    def semantic(cls) -> tuple["MaskName", ...]:
        return cls.INFER_TARGET, cls.FIXED_TARGET, cls.OBSERVED_CONDITION, cls.FIXED_TARGET_VALUE

    @classmethod
    def minimum_batch_rank(cls, name: str) -> int | None:
        """Return the canonical rank at which a mask has a leading batch axis."""
        if name == cls.ATTENTION_MASK:
            return 3
        if name == cls.MASK or name in cls.semantic():
            return 2
        return None
