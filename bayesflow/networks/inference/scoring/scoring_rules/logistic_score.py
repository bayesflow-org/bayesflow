import keras

from bayesflow.types import Shape, Tensor
from bayesflow.utils import weighted_mean
from bayesflow.utils.serialization import serializable

from .scoring_rule import ScoringRule
from .scaled_exponential_score import _pairwise_diff


@serializable("bayesflow.scoring_rules", disable_module_check=True)
class LogisticScore(ScoringRule):
    r"""Logistic scoring rule for amortized Bayes factor estimation.

    For the true model :math:`m`:

    .. math::

        S(\{f_k\}, m) = \sum_{k \neq m} \log\!\left(1 + e^{f_k(x) - f_m(x)}\right)

    The unique minimiser shares the same structure as :class:`ExponentialScore`.
    The public :meth:`~bayesflow.approximators.ModelComparisonApproximator.predict` method
    returns :math:`\log K_{k,0} = \log p(x \mid \mathcal{M}_k) - \log p(x \mid \mathcal{M}_0)`
    (reference in the denominator).
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
        scores = keras.ops.sum(mask * keras.ops.softplus(diff), axis=-1)
        return weighted_mean(scores, weights)

    def get_config(self):
        return super().get_config() | self.config
