import keras

from bayesflow.types import Shape, Tensor
from bayesflow.utils import weighted_mean
from bayesflow.utils.serialization import serializable

from .scoring_rule import ScoringRule


def _pairwise_diff(f: Tensor, targets: Tensor) -> Tensor:
    """Prepend f_0=0 and compute f_k - f_m for all k, where m is the true model."""
    zeros = keras.ops.zeros_like(f[..., :1])
    f_full = keras.ops.concatenate([zeros, f], axis=-1)  # (..., M)
    m = keras.ops.cast(keras.ops.argmax(targets, axis=-1), dtype="int32")
    m_idx = keras.ops.expand_dims(m, axis=-1)  # (..., 1)
    f_m = keras.ops.take_along_axis(f_full, m_idx, axis=-1)  # (..., 1)
    return f_full - f_m  # (..., M), broadcast


@serializable("bayesflow.scoring_rules", disable_module_check=True)
class ExponentialScore(ScoringRule):
    r"""Exponential scoring rule for amortized Bayes factor estimation.

    The network outputs :math:`M - 1` log Bayes factors
    :math:`(f_1, \ldots, f_{M-1})` relative to model 0 (:math:`f_0 = 0`
    by convention).  For the true model :math:`m`:

    :math:`S(\{f_k\}, m) = \sum_{k=0}^{M-1} \exp(f_k - f_m)`

    The loss is minimised when :math:`f_m \gg f_k` for all :math:`k \neq m`,
    i.e. when the log Bayes factor strongly favours the true model.
    """

    NOT_TRANSFORMING_LIKE_VECTOR_WARNING = ("log_bayes_factors",)
    # Small-stddev init keeps initial log-odds near zero, preventing exp() overflow at the start of training.
    _head_kernel_initializer = keras.initializers.TruncatedNormal(mean=0.0, stddev=0.01)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.config = {}

    def get_head_shapes_from_target_shape(self, target_shape: Shape) -> dict[str, Shape]:
        target_shape = tuple(target_shape)
        return dict(log_bayes_factors=target_shape[1:-1] + (target_shape[-1] - 1,))

    def score(self, estimates: dict[str, Tensor], targets: Tensor, weights: Tensor = None) -> Tensor:
        """
        Computes the exponential Bayes factor score.

        Parameters
        ----------
        estimates : dict[str, Tensor]
            Must contain ``"log_bayes_factors"`` of shape ``(..., M-1)``.
        targets : Tensor
            One-hot encoded true model labels of shape ``(..., M)``.
        weights : Tensor, optional
            Per-sample weights for a weighted mean.

        Returns
        -------
        Tensor
            (Optionally weighted) mean exponential score over the batch.
        """
        diff = _pairwise_diff(estimates["log_bayes_factors"], targets)
        mask = 1.0 - targets
        half_diff = diff / 2.0
        # Adjust per-term clip so sum of (M-1) terms stays within float32 range
        M = keras.ops.cast(keras.ops.shape(diff)[-1], dtype="float32")
        clip_max = 88.0 - keras.ops.log(keras.ops.maximum(M - 1.0, 1.0))
        scores = keras.ops.sum(
            mask * keras.ops.exp(keras.ops.minimum(keras.ops.maximum(half_diff, -88.0), clip_max)), axis=-1
        )
        return weighted_mean(scores, weights)

    def get_config(self):
        return super().get_config() | self.config
