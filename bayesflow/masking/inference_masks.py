from collections.abc import Mapping
from dataclasses import dataclass, field

from bayesflow.types import Tensor

from .mask_name import MaskName


@dataclass(frozen=True)
class InferenceMasks:
    """Masks prepared for one inference-network operation."""

    fixed_target_mask: Tensor | None = None
    infer_target_mask: Tensor | None = None
    observed_condition_mask: Tensor | None = None
    fixed_target_value: Tensor | None = None
    subnet_kwargs: Mapping[str, Tensor | None] = field(default_factory=dict)

    def semantic_kwargs(self) -> dict[str, Tensor]:
        return {name.value: value for name in MaskName.semantic() if (value := getattr(self, name.value)) is not None}
