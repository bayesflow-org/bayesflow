import keras
from bayesflow.types import Tensor
from .permutations import FixedPermutation


def skip_fixed_dims(inputs: Tensor, outputs: Tensor, log_jac: Tensor, mask: Tensor | None) -> tuple[Tensor, Tensor]:
    """Keep fixed dims at their input value and zero their log-Jacobian."""
    if mask is not None:
        outputs = keras.ops.where(mask, outputs, inputs)
        log_jac = keras.ops.where(mask, log_jac, 0.0)
    return outputs, log_jac


def split_mask(mask: Tensor | None, pivot: int) -> tuple[Tensor | None, Tensor | None]:
    if mask is None:
        return None, None
    return mask[..., :pivot], mask[..., pivot:]


def trace_permutations(layers: list[keras.Layer], xz: Tensor | None) -> list[Tensor | None]:
    if xz is None:
        return [None] * (len(layers) + 1)
    trace = [xz]
    for layer in layers:
        if isinstance(layer, FixedPermutation):
            xz = keras.ops.take(xz, layer.forward_indices, axis=-1)
        trace.append(xz)
    return trace


def permute_like(layers: list[keras.Layer], xz: Tensor) -> Tensor:
    return trace_permutations(layers, xz)[-1]
