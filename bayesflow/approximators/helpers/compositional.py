from collections.abc import Callable

import numpy as np
import keras

from bayesflow.adapters import Adapter
from bayesflow.types import Tensor


def prepare_compute_prior_score(
    samples: Tensor,
    time: Tensor,
    compute_prior_score: Callable[[dict[str, np.ndarray], Tensor], dict[str, np.ndarray]],
    adapter: Adapter,
    standardizer,
) -> Tensor:
    samples = keras.tree.map_structure(
        lambda s: standardizer.maybe_standardize(s, key="inference_variables", stage="inference", forward=False),
        samples,
    )
    if keras.backend.backend() != "torch":  # samples cannot be converted to numpy, otherwise it breaks
        adapted_samples, log_det_jac = keras.tree.map_structure(
            lambda s: adapter({"inference_variables": s}, inverse=True, strict=False, log_det_jac=True),
            samples,
        )
    else:  # samples need to be converted to numpy, adapter cannot use torch tensors
        adapted_samples, log_det_jac = keras.tree.map_structure(
            lambda s: adapter(
                {"inference_variables": keras.ops.convert_to_numpy(s)}, inverse=True, strict=False, log_det_jac=True
            ),
            samples,
        )

    if len(log_det_jac) > 0:
        problematic_keys = [key for key in log_det_jac if log_det_jac[key] != 0.0]
        raise NotImplementedError(
            f"Cannot use compositional sampling with adapters "
            f"that have non-zero log_det_jac. Problematic keys: {problematic_keys}"
        )

    prior_score = compute_prior_score(adapted_samples, time)
    for key in adapted_samples:
        prior_score[key] = keras.ops.cast(prior_score[key], "float32")

    prior_score = keras.tree.map_structure(keras.ops.convert_to_tensor, prior_score)
    out = keras.ops.concatenate([prior_score[key] for key in adapted_samples], axis=-1)

    if "inference_variables" in standardizer.standardize:
        # Apply jacobian correction from standardization
        # For standardization T^{-1}(z) = z * std + mean, the jacobian is diagonal with std on diagonal
        # The gradient of log|det(J)| w.r.t. z is 0 since log|det(J)| = sum(log(std)) is constant w.r.t. z
        # But we need to transform the score: score_z = score_x * std where x = T^{-1}(z)
        standardize_layer = standardizer.standardize_layers["inference_variables"]

        # Compute the correct standard deviation for all components
        std_components = []
        for idx in range(len(standardize_layer.moving_mean)):
            std_val = standardize_layer.moving_std(idx)
            std_components.append(std_val)

        # Concatenate std components to match the shape of out
        if len(std_components) == 1:
            std = std_components[0]
        else:
            std = keras.ops.concatenate(std_components, axis=-1)

        # Expand std to match batch dimension of out
        std_expanded = keras.ops.expand_dims(std, 0)  # Add batch dimensions

        # Apply the jacobian: score_z = score_x * std
        out = out * std_expanded
    return out
