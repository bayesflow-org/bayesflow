from __future__ import annotations

import inspect
from collections.abc import Callable, Iterable, Mapping
from dataclasses import replace
from typing import Any

import keras

from bayesflow.types import Tensor
from bayesflow.utils.serialization import serializable
from bayesflow.utils.tensor_utils import repeat_and_flatten

from .inference_masks import InferenceMasks
from .mask_name import MaskName


@serializable("bayesflow.masking")
class MaskHandler:
    """Generate semantic masks and route masks between BayesFlow interfaces.

    Semantic masks change the target distribution. For example, masking missing
    targets lets a network trained on :math:`p(x_1, x_2)` learn its marginals.
    Padding and attention masks are structural; they support operations such as
    learning filtering distributions with an ``AutoregressiveEstimator``.

    Parameters
    ----------
    fixed_target_prob : float, optional
        Probability of fixing a target coordinate during training.
    missing_target_prob : float, optional
        Probability of hiding a fixed target coordinate during training.
    missing_conditions_prob : float, optional
        Probability of hiding a condition coordinate during training.
    seed : int, optional
        Seed used to initialize the local random generator for sampling training
        masks.
    """

    summary_mask_routes = {
        MaskName.SUMMARY_ATTENTION: MaskName.ATTENTION_MASK,
        MaskName.SUMMARY: MaskName.MASK,
    }
    inference_mask_routes = {
        MaskName.INFERENCE_ATTENTION: MaskName.ATTENTION_MASK,
        MaskName.INFERENCE: MaskName.MASK,
    }
    subnet_mask_names = (
        MaskName.ATTENTION_MASK,
        MaskName.MASK,
        MaskName.FIXED_TARGET,
        MaskName.INFER_TARGET,
        MaskName.OBSERVED_CONDITION,
    )

    def __init__(
        self,
        *,
        fixed_target_prob: float = 0.0,
        missing_target_prob: float = 0.0,
        missing_conditions_prob: float = 0.0,
        seed: int | None = None,
    ):
        self.fixed_target_prob = fixed_target_prob
        self.missing_target_prob = missing_target_prob
        self.missing_conditions_prob = missing_conditions_prob
        self.seed = seed
        self.seed_generator = keras.random.SeedGenerator(seed)

    def get_config(self) -> dict[str, Any]:
        return {
            "fixed_target_prob": self.fixed_target_prob,
            "missing_target_prob": self.missing_target_prob,
            "missing_conditions_prob": self.missing_conditions_prob,
            "seed": self.seed,
        }

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> MaskHandler:
        return cls(**config)

    @staticmethod
    def _collect(routes: Mapping[str, str] | Iterable[str], *sources: Mapping[str, Any]) -> dict[str, Any]:
        """Route non-``None`` values, with later sources taking precedence."""
        if not isinstance(routes, Mapping):
            routes = {name: name for name in routes}

        return {
            target: source[source_name]
            for source in sources
            for source_name, target in routes.items()
            if source.get(source_name) is not None
        }

    @classmethod
    def collect_summary_masks(cls, *sources: Mapping[str, Any]) -> dict[str, Any]:
        """Map summary-boundary mask names to their subnet names."""
        return cls._collect(cls.summary_mask_routes, *sources)

    @classmethod
    def collect_inference_masks(cls, *sources: Mapping[str, Any]) -> dict[str, Any]:
        """Map inference-boundary mask names to their subnet names."""
        return cls._collect(cls.inference_mask_routes, *sources)

    @classmethod
    def collect_semantic_masks(cls, *sources: Mapping[str, Any]) -> dict[str, Any]:
        """Collect semantic masks without renaming them."""
        return cls._collect(MaskName.semantic(), *sources)

    @classmethod
    def collect_mask_kwargs(cls, *sources: Mapping[str, Any]) -> dict[str, Any]:
        """Collect canonical subnet masks and the fixed target value."""
        return cls._collect((*cls.subnet_mask_names, MaskName.FIXED_TARGET_VALUE), *sources)

    @classmethod
    def collect_subnet_masks(cls, subnet: keras.Layer, *sources: Mapping[str, Any]) -> dict[str, Any]:
        """Collect masks explicitly named in ``subnet.call``."""
        masks = cls._collect(cls.subnet_mask_names, *sources)
        parameters = inspect.signature(subnet.call).parameters
        return {name: value for name, value in masks.items() if name in parameters}

    @classmethod
    def with_inference_masks(
        cls,
        kwargs: Mapping[str, Any],
        adapted: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Add masks discovered in adapted data to inference call kwargs.

        Explicit kwargs take precedence over masks found in adapted data.
        Boundary names such as ``inference_mask`` are replaced by the canonical
        subnet names ``mask`` and ``attention_mask``.
        """
        adapted = adapted or {}
        passthrough = cls.without_semantic_kwargs(kwargs)
        for name in cls.inference_mask_routes:
            passthrough.pop(name, None)
        return cls.collect_inference_masks(adapted, kwargs) | passthrough | cls.collect_semantic_masks(adapted, kwargs)

    @classmethod
    def without_semantic_kwargs(cls, source: Mapping[str, Any]) -> dict[str, Any]:
        """Return kwargs without semantic masks or fixed target values."""
        names = MaskName.semantic()
        return {name: value for name, value in source.items() if name not in names}

    @classmethod
    def without_mask_kwargs(cls, source: Mapping[str, Any]) -> dict[str, Any]:
        """Return kwargs without canonical subnet or semantic mask fields."""
        names = (*cls.subnet_mask_names, MaskName.FIXED_TARGET_VALUE)
        return {name: value for name, value in source.items() if name not in names}

    @classmethod
    def reject_structural_masks(cls, *sources: Mapping[str, Any], reason: str) -> None:
        """Raise if an attention or padding mask would be ignored."""
        for name in cls._collect((MaskName.ATTENTION_MASK, MaskName.MASK), *sources):
            raise ValueError(f"`{name}` would be ignored: {reason}.")

    @staticmethod
    def maybe_apply(value: Tensor, mask: Tensor | None, replacement: Tensor | None = None) -> Tensor:
        """Keep ``value`` where ``mask`` is one and use ``replacement`` where it is zero.
        If no replacement is provided, a zero replacement will be used."""
        if mask is None:
            return value
        if replacement is None:
            replacement = keras.ops.zeros_like(value)
        return keras.ops.where(keras.ops.cast(mask, "bool"), value, replacement)

    @staticmethod
    def per_token_target_mask(mask: Tensor | None, targets: Tensor) -> Tensor | None:
        """Reshape a ``(num_targets,)`` or ``(batch, num_targets)`` target mask to ``(batch, num_targets, 1)``,
        one value per target token of a transformer that broadcasts over the token width."""
        if mask is None:
            return None

        mask = keras.ops.cast(mask, targets.dtype)
        if len(mask.shape) == 1:
            mask = keras.ops.expand_dims(mask, axis=0)
        return keras.ops.expand_dims(mask, axis=-1)

    @classmethod
    def has_batch_axis(
        cls,
        name: str,
        value: Tensor,
        batch_size: int | None,
        *,
        target_structural_rank: int = 0,
        condition_structural_rank: int = 0,
    ) -> bool:
        """Return whether ``value`` carries the leading data-batch axis.

        Padding masks use shared ``(S,)`` and batched ``(B, S)`` shapes;
        attention masks use shared ``(T, S)`` and batched ``(B, T, S)``
        shapes. Semantic masks additionally account for the structure of the
        tensor they mask. If a shape is also valid without a batch axis, a
        matching leading batch dimension takes precedence.
        """
        minimum_rank = MaskName.minimum_batch_rank(name)
        if batch_size is None or len(value.shape) == 0:
            return False
        if minimum_rank is None:
            leading_dim = value.shape[0]
            return leading_dim is not None and int(leading_dim) == int(batch_size)
        if name == MaskName.OBSERVED_CONDITION:
            minimum_rank += condition_structural_rank
        elif name in (MaskName.FIXED_TARGET, MaskName.INFER_TARGET, MaskName.FIXED_TARGET_VALUE):
            minimum_rank += target_structural_rank
        if len(value.shape) < minimum_rank:
            return False
        leading_dim = value.shape[0]
        return leading_dim is not None and int(leading_dim) == int(batch_size)

    @classmethod
    def batched_keys(
        cls,
        source: Mapping[str, Any],
        batch_size: int | None,
        target_structural_rank: int = 0,
        condition_structural_rank: int = 0,
    ) -> set[str]:
        """Return tensor fields that carry the leading data-batch axis."""
        return {
            name
            for name, value in source.items()
            if hasattr(value, "shape")
            and cls.has_batch_axis(
                name,
                value,
                batch_size,
                target_structural_rank=target_structural_rank,
                condition_structural_rank=condition_structural_rank,
            )
        }

    @classmethod
    def infer_batch_size(cls, source: Mapping[str, Any], target_structural_rank: int = 0) -> int | None:
        """Infer the batch size from a semantic target mask or fixed value."""
        for name in (MaskName.FIXED_TARGET, MaskName.FIXED_TARGET_VALUE, MaskName.INFER_TARGET):
            value = source.get(name)
            if value is None or not hasattr(value, "shape"):
                continue
            if (
                len(value.shape) >= MaskName.minimum_batch_rank(name) + target_structural_rank
                and value.shape[0] is not None
            ):
                return value.shape[0]
        return None

    @classmethod
    def select_autoregressive_step(
        cls,
        name: str,
        value: Tensor,
        *,
        step: int,
        num_steps: int,
        has_batch_axis: bool,
    ) -> Tensor:
        """Select one autoregressive step when ``value`` has a step axis.

        Shared step axes use axis 0 and batched step axes use axis 1. Singleton
        step axes are reused, while event-only values are returned unchanged.
        """
        time_axis = 1 if has_batch_axis else 0
        minimum_batch_rank = MaskName.minimum_batch_rank(name)
        if minimum_batch_rank is None:
            if len(value.shape) < time_axis + 2:
                return value
        else:
            event_rank = minimum_batch_rank - 1
            structural_rank = len(value.shape) - event_rank - int(has_batch_axis)
            if structural_rank < 1:
                return value

        time_size = value.shape[time_axis]
        if time_size == 1:
            return keras.ops.take(value, 0, axis=time_axis)
        if time_size == num_steps:
            return keras.ops.take(value, step, axis=time_axis)
        return value

    def resolve(
        self,
        *sources: Mapping[str, Any],
        targets: Tensor,
        conditions: Tensor | None = None,
        stage: str = "inference",
        subnet: keras.Layer | None = None,
    ) -> InferenceMasks:
        """Resolve semantic masks for one inference-network operation.

        Parameters
        ----------
        *sources : Mapping[str, Any]
            Mappings containing explicit semantic masks. Later mappings take
            precedence over earlier ones.
        targets : Tensor
            Target tensor whose shape is used to broadcast target masks and
            fixed target values.
        conditions : Tensor, optional
            Condition tensor whose shape is used to broadcast the observed
            condition mask.
        stage : str, optional
            Current execution stage. Probability-based masks are sampled only
            during ``"training"``; other stages use explicit masks only.
        subnet : keras.Layer, optional
            Mask consumer. Generated ``infer_target_mask`` and
            ``observed_condition_mask`` values are limited to masks supported
            by the subnet. ``subnet_kwargs`` in the result contains the masks
            declared by its ``call`` signature.

        Returns
        -------
        InferenceMasks
            Broadcast semantic masks and the masks to pass to it, if
            ``subnet`` is supplied.

        Raises
        ------
        ValueError
            If ``subnet`` is supplied and an ``attention_mask`` or ``mask``
            in ``sources`` is not named in its ``call`` signature.
        """
        training = stage == "training"
        provided = self.collect_semantic_masks(*sources)
        dtype = keras.config.floatx()

        def supported(name):
            return subnet is None or name in inspect.signature(subnet.call).parameters

        if subnet is not None:
            for name in self._collect((MaskName.ATTENTION_MASK, MaskName.MASK), *sources):
                if not supported(name):
                    raise ValueError(f"`{name}` would be ignored: `{type(subnet).__name__}.call` does not name it.")

        fixed_target_mask = provided.get(MaskName.FIXED_TARGET)
        infer_target_mask = provided.get(MaskName.INFER_TARGET)

        if fixed_target_mask is None and training and self.fixed_target_prob > 0.0:
            fixed_target_mask = self.random_mask(
                targets,
                drop_prob=self.fixed_target_prob,
                keep_one=True,
                eligible=infer_target_mask,
            )
        elif fixed_target_mask is not None:
            fixed_target_mask = keras.ops.broadcast_to(
                keras.ops.cast(fixed_target_mask, dtype),
                keras.ops.shape(targets),
            )

        if (
            infer_target_mask is None
            and fixed_target_mask is not None
            and training
            and self.missing_target_prob > 0.0
            and supported(MaskName.INFER_TARGET)
        ):
            infer_target_mask = keras.ops.maximum(
                fixed_target_mask,
                self.random_mask(targets, drop_prob=self.missing_target_prob),
            )
        elif infer_target_mask is not None:
            infer_target_mask = keras.ops.broadcast_to(
                keras.ops.cast(infer_target_mask, dtype),
                keras.ops.shape(targets),
            )

        if infer_target_mask is not None:
            fixed_target_mask = (
                infer_target_mask if fixed_target_mask is None else fixed_target_mask * infer_target_mask
            )

        observed_condition_mask = provided.get(MaskName.OBSERVED_CONDITION)
        if (
            observed_condition_mask is None
            and training
            and self.missing_conditions_prob > 0.0
            and supported(MaskName.OBSERVED_CONDITION)
        ):
            if conditions is not None:
                observed_condition_mask = self.random_mask(conditions, drop_prob=self.missing_conditions_prob)
        elif observed_condition_mask is not None and conditions is not None:
            observed_condition_mask = keras.ops.broadcast_to(
                keras.ops.cast(observed_condition_mask, dtype),
                keras.ops.shape(conditions),
            )

        fixed_target_value = provided.get(MaskName.FIXED_TARGET_VALUE)
        if fixed_target_value is not None:
            fixed_target_value = keras.ops.broadcast_to(
                keras.ops.cast(fixed_target_value, targets.dtype),
                keras.ops.shape(targets),
            )

        masks = InferenceMasks(
            fixed_target_mask=fixed_target_mask,
            infer_target_mask=infer_target_mask,
            observed_condition_mask=observed_condition_mask,
            fixed_target_value=fixed_target_value,
        )
        if subnet is not None:
            subnet_kwargs = self.collect_subnet_masks(subnet, *sources, masks.semantic_kwargs())
            masks = replace(masks, subnet_kwargs=subnet_kwargs)
        return masks

    def resolve_compositional(
        self,
        subnet: keras.Layer,
        *sources: Mapping[str, Any],
        targets: Tensor,
        conditions: Tensor,
    ) -> InferenceMasks:
        """Resolve compositional masks, defaulting every condition to observed."""
        masks = self.resolve(*sources, targets=targets, conditions=conditions)
        if masks.observed_condition_mask is None:
            masks = replace(masks, observed_condition_mask=keras.ops.ones_like(conditions))
        subnet_kwargs = self.collect_subnet_masks(subnet, *sources, masks.semantic_kwargs())
        return replace(masks, subnet_kwargs=subnet_kwargs)

    def repeat_compositional(self, masks: InferenceMasks, num_repeats: int) -> InferenceMasks:
        """Repeat resolved masks before flattening compositional inputs.

        Semantic masks must already be broadcast to their target or condition
        tensors. The condition mask is attached after compositional selection.
        """
        semantic_kwargs = self.collect_semantic_masks(masks.subnet_kwargs)
        subnet_kwargs = {}
        for name, value in masks.subnet_kwargs.items():
            if name in semantic_kwargs:
                continue
            minimum_batch_rank = MaskName.minimum_batch_rank(name)
            has_batch_axis = (
                minimum_batch_rank is not None
                and hasattr(value, "shape")
                and len(value.shape) >= minimum_batch_rank
                and value.shape[0] != 1
            )
            subnet_kwargs[name] = repeat_and_flatten(value, num_repeats) if has_batch_axis else value

        repeated = {
            name: repeat_and_flatten(value, num_repeats)
            for name, value in masks.semantic_kwargs().items()
            if name != MaskName.OBSERVED_CONDITION
        }
        repeated_masks = replace(masks, **repeated, observed_condition_mask=None)
        for name, value in repeated_masks.semantic_kwargs().items():
            if name in semantic_kwargs:
                subnet_kwargs[name] = value
        if MaskName.OBSERVED_CONDITION in semantic_kwargs:
            subnet_kwargs[MaskName.OBSERVED_CONDITION] = None
        return replace(repeated_masks, subnet_kwargs=subnet_kwargs)

    @staticmethod
    def with_flattened_condition_mask(
        masks: InferenceMasks,
        observed_condition_mask: Tensor,
    ) -> InferenceMasks:
        """Flatten and attach a selected compositional condition mask."""
        condition_shape = tuple(keras.ops.shape(observed_condition_mask)[2:])
        observed_condition_mask = keras.ops.reshape(
            observed_condition_mask,
            (-1,) + condition_shape,
        )
        subnet_kwargs = masks.subnet_kwargs
        if MaskName.OBSERVED_CONDITION in subnet_kwargs:
            subnet_kwargs = subnet_kwargs | {MaskName.OBSERVED_CONDITION: observed_condition_mask}
        return replace(
            masks,
            observed_condition_mask=observed_condition_mask,
            subnet_kwargs=subnet_kwargs,
        )

    def integrator_callbacks(
        self,
        masks: InferenceMasks,
        replacement: Tensor | None = None,
        state_key: str = "xz",
    ) -> tuple[Callable | None, Callable | None]:
        """Build state and update projections that preserve fixed targets."""
        if masks.fixed_target_mask is None:
            return None, None

        def project_state(state):
            fixed_value = masks.fixed_target_value if replacement is None else replacement
            return state | {
                state_key: self.maybe_apply(state[state_key], masks.fixed_target_mask, fixed_value),
            }

        def project_updates(state):
            return state | {state_key: self.maybe_apply(state[state_key], masks.fixed_target_mask)}

        return project_state, project_updates

    def random_mask(
        self,
        reference: Tensor,
        *,
        drop_prob: float,
        keep_one: bool = False,
        eligible: Tensor | None = None,
    ) -> Tensor:
        """Sample a binary keep mask with the shape of ``reference``.

        ``eligible`` limits coordinates that may be kept. If ``keep_one`` is
        true, at least one eligible coordinate on the final axis remains one
        whenever that axis contains an eligible coordinate.
        """
        dtype = keras.config.floatx()
        random_values = keras.random.uniform(
            shape=keras.ops.shape(reference),
            dtype=dtype,
            seed=self.seed_generator,
        )
        mask = keras.ops.cast(random_values > drop_prob, dtype)

        if eligible is not None:
            eligible = keras.ops.cast(eligible, dtype)
            mask = mask * eligible

        if keep_one:
            candidates = random_values
            if eligible is not None:
                candidates = keras.ops.where(
                    keras.ops.cast(eligible, "bool"),
                    random_values,
                    keras.ops.full_like(random_values, -1.0),
                )
            keep = keras.ops.cast(candidates >= keras.ops.max(candidates, axis=-1, keepdims=True), dtype)
            if eligible is not None:
                has_eligible = keras.ops.any(keras.ops.cast(eligible, "bool"), axis=-1, keepdims=True)
                keep = keep * eligible * keras.ops.cast(has_eligible, dtype)
            mask = keras.ops.maximum(mask, keep)

        return mask
