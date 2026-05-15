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
        # sum softplus(diff) over k != m; mask out k == m via (1 - targets)
        mask = 1.0 - targets  # 0 at true model, 1 elsewhere
        scores = keras.ops.sum(mask * keras.ops.softplus(diff), axis=-1)
        return weighted_mean(scores, weights)

    def get_config(self):
        return super().get_config() | self.config
