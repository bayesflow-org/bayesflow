"""Compatibility imports for estimator helpers moved to :mod:`bayesflow.estimators`."""

import keras

from bayesflow.estimators.helpers import (
    AutoregressiveConditionBuilder,
    AutoregressiveSampler,
    ConditionBuilder,
    Sampler,
    build_prior_score_fn,
)


# Preserve deserialization of models saved before the package rename without
# changing the canonical registered names used for newly serialized objects.
keras.saving.get_custom_objects().update(
    {
        "bayesflow.approximators>ConditionBuilder": ConditionBuilder,
        "bayesflow.approximators>AutoregressiveConditionBuilder": AutoregressiveConditionBuilder,
        "bayesflow.approximators>Sampler": Sampler,
        "bayesflow.approximators>AutoregressiveSampler": AutoregressiveSampler,
    }
)

__all__ = [
    "ConditionBuilder",
    "AutoregressiveConditionBuilder",
    "Sampler",
    "AutoregressiveSampler",
    "build_prior_score_fn",
]
