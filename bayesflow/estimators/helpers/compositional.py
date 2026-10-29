from collections.abc import Callable
import inspect

import keras

from bayesflow._backend import vjp
from bayesflow.adapters import Adapter
from bayesflow.adapters.transforms import Concatenate, ConvertDType, FilterTransform, Rename, ToArray
from bayesflow.types import Tensor


def build_prior_score_fn(
    compute_prior_score: Callable,
    adapter: Adapter,
    standardizer,
) -> Callable[[Tensor, Tensor], Tensor]:
    """Build a prior score step function with all fixed state captured once.

    Inspects ``compute_prior_score`` signature, pre-computes the standardization std
    correction, and returns a ``(samples, time) -> Tensor`` closure.

    The user supplies the prior score in the original parameter space, while the diffusion
    samples in the adapted (and standardized) space. Relating the two is a change of variables
    over the inverse adapter ``theta = T(z)``::

        log p_Z(z) = log p_Theta(T(z)) + log |det dT/dz|

    so that the score in the sampling space is::

        score_z = J^T score_theta + grad_z log |det J|,   J = dT/dz

    Parameters
    ----------
    compute_prior_score : Callable
        Function that computes prior scores from backend tensors.  May or may not accept a
        ``time`` keyword argument.
    adapter : Adapter
        Adapter used to perform inverse transformation of samples. Its transforms have to be
        differentiable, since the change of variables above is taken through them.
    standardizer : object
        Fitted standardizer with ``standardize`` dict and ``standardize_layers``.

    Returns
    -------
    Callable[[Tensor, Tensor], Tensor]
        Step function with signature ``(samples, time) -> Tensor``.
    """

    # Capture fixed states
    prior_has_time = "time" in inspect.signature(compute_prior_score).parameters

    # gradients must flow through the transforms for the change of variables below
    differentiable_adapter = Adapter(adapter.transforms, device=adapter.device, differentiable=True)

    if "inference_variables" in standardizer.standardize:
        standardize_layer = standardizer.standardize_layers["inference_variables"]
        num_components = len(standardize_layer.moving_mean)
        std = keras.ops.concatenate([standardize_layer.moving_std(idx) for idx in range(num_components)], axis=-1)
        mean = keras.ops.concatenate([standardize_layer.moving_mean[idx] for idx in range(num_components)], axis=-1)
    else:
        std = mean = None

    # adapters that only move values around have a permutation Jacobian (J^T = J^-1, log |det J| = 0),
    permutation_adapter = all(_is_permutation(transform) for transform in adapter.transforms)

    def _prior_score(adapted_samples: dict[str, Tensor], time: Tensor) -> dict[str, Tensor]:
        if prior_has_time:
            prior_score = compute_prior_score(adapted_samples, time=time)
        else:
            prior_score = compute_prior_score(adapted_samples)
        return {key: keras.ops.cast(prior_score[key], keras.config.floatx()) for key in adapted_samples}

    def _adapter_prior_score(samples: Tensor, time: Tensor) -> Tensor:
        """Adapter inverse, change of variables, and the user-provided prior score."""
        if permutation_adapter:
            adapted_samples = differentiable_adapter({"inference_variables": samples}, inverse=True, strict=False)
            prior_score = _prior_score(adapted_samples, time)
            return differentiable_adapter(prior_score, strict=False)["inference_variables"]

        def inverse_with_log_det_jac(z: Tensor):
            return differentiable_adapter({"inference_variables": z}, inverse=True, strict=False, log_det_jac=True)

        (adapted_samples, log_det_jac), vjp_fn = vjp(inverse_with_log_det_jac, samples)

        cotangent = (
            _prior_score(adapted_samples, time),
            # log |det J| enters log p_Z with unit weight
            {key: keras.ops.ones_like(value) for key, value in log_det_jac.items()},
        )

        return vjp_fn(cotangent)[0]

    def _step(samples: Tensor, time: Tensor) -> Tensor:
        # unstandardize with the moments captured above
        if std is not None:
            samples = samples * std + mean

        out = _adapter_prior_score(samples, time)

        if not prior_has_time:
            out = (1 - time) * out

        # Apply Jacobian correction from standardization:
        # For T^{-1}(z) = z * std + mean the score transforms as score_z = score_x * std
        if std is not None:
            out = out * std

        return out

    return _step


def _is_permutation(transform) -> bool:
    """Whether ``transform`` only renames, concatenates, or casts values, i.e., has a permutation Jacobian."""
    if isinstance(transform, FilterTransform):
        return issubclass(transform.transform_constructor, (ToArray, ConvertDType))
    return isinstance(transform, (Rename, Concatenate))
