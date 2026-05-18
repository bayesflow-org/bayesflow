import keras

from bayesflow.types import Shape, Tensor
from bayesflow.utils import weighted_mean
from bayesflow.utils.serialization import serializable

from .scoring_rule import ScoringRule
from .exponential_score import _pairwise_diff


@serializable("bayesflow.scoring_rules", disable_module_check=True)
class LogisticScore(ScoringRule):
    r"""Logistic scoring rule for amortized Bayes factor estimation.

    The network outputs :math:`M - 1` log Bayes factors
    :math:`(f_1, \ldots, f_{M-1})` relative to model 0 (:math:`f_0 = 0`).
    For the true model :math:`m`:

    :math:`S(\{f_k\}, m) = \sum_{k \neq m} \log\!\left(1 + \exp(f_k - f_m)\right)`

    Equivalent to applying the binary logistic loss to every pairwise
    log-odds :math:`f_k - f_m`.  Compared to :class:`ExponentialScore`, the
    log link makes it less sensitive to large Bayes factors.
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
        Computes the logistic Bayes factor score.

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
            (Optionally weighted) mean logistic score over the batch.
        """
        diff = _pairwise_diff(estimates["log_bayes_factors"], targets)
        mask = 1.0 - targets
        half_diff = diff / 2.0
        # log(1 + s^2) = softplus(2*log_s); compute log_s via masked logsumexp to avoid s^2 overflow
        masked_hd = keras.ops.where(mask > 0.5, half_diff, keras.ops.full_like(half_diff, -1e9))
        max_hd = keras.ops.max(masked_hd, axis=-1, keepdims=True)
        log_s = max_hd[..., 0] + keras.ops.log(
            keras.ops.sum(mask * keras.ops.exp(keras.ops.clip(half_diff - max_hd, -88.0, 0.0)), axis=-1)
        )
        scores = keras.ops.softplus(2.0 * log_s)
        return weighted_mean(scores, weights)

    def get_config(self):
        return super().get_config() | self.config
