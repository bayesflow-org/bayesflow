from enum import StrEnum

import keras

from bayesflow.types import Tensor, Shape
from .keras_utils import call_accepts_kwarg


class MaskName(StrEnum):
    """Canonical kwarg names for the diffusion-type input-masking scheme.

    These identify the three optional masks that diffusion-type inference networks
    (diffusion, flow matching, consistency).

    INFER_TARGET :
        ``1`` = infer, ``0`` =  marginalized out.
         Governed by ``missing_target_prob``.
    FIXED_TARGET :
        ``1`` = infer/noised, ``0`` = conditioned on fixed value.
        Governed by``fixed_target_prob``.
    OBSERVED_CONDITION :
        ``1`` = present, ``0`` = missing/marginalized out.
        Governed by ``missing_conditions_prob``.
    FIXED_TARGET_VALUE :
        Value of the fixed target.
    """

    INFER_TARGET = "infer_target_mask"
    FIXED_TARGET = "fixed_target_mask"
    OBSERVED_CONDITION = "observed_condition_mask"
    FIXED_TARGET_VALUE = "fixed_target_value"


def maybe_mask_tensor(data: Tensor, mask: Tensor | None = None, replacement: Tensor = None) -> Tensor:
    """Apply a binary mask to a tensor if a mask is passed, blending with a replacement where masked.

    Parameters
    ----------
    data : Tensor
        The tensor to mask.
    mask : Tensor or None, optional
        Binary mask where 1.0 = keep, 0.0 = replace. If ``None``, *data* is
        returned unchanged.
    replacement : Tensor, optional
        Values to use where the mask is 0. If ``None``, zeros are used.

    Returns
    -------
    Tensor
        ``mask * data + (1 - mask) * replacement``, or *data* when *mask* is
        ``None``.
    """
    if mask is None:
        return data

    if replacement is None:
        return mask * data

    return mask * data + (1 - mask) * replacement


def random_mask(
    shape: Shape, drop_prob: float, keep_one: bool = False, seed_generator: keras.random.SeedGenerator = None
) -> Tensor | float:
    """Generate an element-wise random mask.

    Each element is independently drawn as 1 (keep) with probability
    ``1 - drop_prob`` and 0 (drop) with probability ``drop_prob``.

    Parameters
    ----------
    shape : Shape
        Shape of the mask to generate.
    drop_prob : float
        Probability of dropping each element. Must be in ``[0, 1]``.
    keep_one : bool
        If True, ensures that at least one element is kept along the last axis.
    seed_generator : keras.random.SeedGenerator, optional
        Seed generator used for randomness.

    Returns
    -------
    Tensor or float
        A mask tensor of the given *shape*, or ``1.0`` when *drop_prob* <= 0.
    """
    if drop_prob <= 0:
        return 1.0

    dtype = keras.backend.floatx()
    random_vals = keras.random.uniform(shape=shape, dtype=dtype, seed=seed_generator)
    mask = keras.ops.cast(random_vals > drop_prob, dtype=dtype)
    if keep_one:
        # Force the largest draw in each last-axis row to be kept, so no row is ever fully dropped.
        mask_keep_one = keras.ops.cast(random_vals >= keras.ops.max(random_vals, axis=-1, keepdims=True), dtype=dtype)
        mask = keras.ops.maximum(mask, mask_keep_one)
    return mask


def sample_input_masks(
    subnet: keras.Layer,
    x: Tensor,
    conditions: Tensor | None,
    subnet_kwargs: dict,
    training: bool,
    fixed_target_prob: float,
    missing_target_prob: float,
    missing_conditions_prob: float,
    seed_generator: keras.random.SeedGenerator = None,
) -> tuple[Tensor | float, Tensor | float, dict]:
    """Generate the target and condition masks and populate ``subnet_kwargs``.

    The diffusion type inference networks (diffusion, flow matching, consistency) share
    the same input-masking scheme: ``fixed_target_prob`` randomly fixes some targets to
    their known value (so the network can condition on them), ``missing_target_prob``
    randomly marks fixed targets as missing, and ``missing_conditions_prob`` randomly marks
    conditions as missing, so the network learns to handle missing fields. The masks are
    forwarded to subnets that accept them (e.g. ``diffusion_transformer``).

    Returns
    -------
    mask_x : Tensor or float
        The per-target inference mask (``1`` = inferred/noised, ``0`` = fixed).
    loss_mask : Tensor or float
        The mask the caller applies to the loss.
    subnet_kwargs : dict
        The keyword arguments, with mask entries added for subnets that accept them.
    """
    mask_x = random_mask(keras.ops.shape(x), fixed_target_prob, keep_one=True, seed_generator=seed_generator)
    if (
        not isinstance(mask_x, float)
        and MaskName.FIXED_TARGET not in subnet_kwargs
        and call_accepts_kwarg(subnet.call, MaskName.FIXED_TARGET)
    ):
        subnet_kwargs[MaskName.FIXED_TARGET] = mask_x

    loss_mask = mask_x
    if training and missing_target_prob > 0:
        if MaskName.INFER_TARGET not in subnet_kwargs and call_accepts_kwarg(subnet.call, MaskName.INFER_TARGET):
            # Only fixed targets (mask_x == 0) may be marked missing -> loss_mask stays equal to mask_x
            raw_missing = random_mask(keras.ops.shape(x), missing_target_prob, seed_generator=seed_generator)
            infer_target_mask = keras.ops.maximum(mask_x, raw_missing)
            subnet_kwargs[MaskName.INFER_TARGET] = infer_target_mask

    if training and missing_conditions_prob > 0 and conditions is not None:
        if MaskName.OBSERVED_CONDITION not in subnet_kwargs and call_accepts_kwarg(
            subnet.call, MaskName.OBSERVED_CONDITION
        ):
            subnet_kwargs[MaskName.OBSERVED_CONDITION] = random_mask(
                keras.ops.shape(conditions), missing_conditions_prob, seed_generator=seed_generator
            )
    return mask_x, loss_mask, subnet_kwargs


def feature_mask(mask: Tensor | None, x: Tensor) -> Tensor | None:
    """Convert a feature mask to shape ``(batch, features, 1)``.

    Parameters
    ----------
    mask : Tensor or None
        A binary feature mask. Supports unbatched masks of shape ``(features,)``,
        batched masks of shape ``(batch, features)``, and masks with a trailing
        singleton or feature axis.
    x : Tensor
        Reference tensor whose dtype and rank determine the returned mask shape.

    Returns
    -------
    Tensor or None
        The mask cast to ``x.dtype`` and expanded to ``(batch, features, 1)``,
        or ``None`` if no mask is passed.
    """
    if mask is None:
        return None

    mask = keras.ops.cast(mask, x.dtype)
    if len(mask.shape) == len(x.shape) and mask.shape[-1] == 1:
        mask = keras.ops.squeeze(mask, axis=-1)
    elif len(mask.shape) == len(x.shape) and len(mask.shape) > 2:
        mask = keras.ops.max(mask, axis=-1)

    if len(mask.shape) == 1:
        mask = keras.ops.expand_dims(mask, axis=0)

    return keras.ops.expand_dims(mask, axis=-1)
