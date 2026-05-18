import keras

from bayesflow.types import Shape, Tensor
from bayesflow.utils import weighted_mean
from bayesflow.utils.serialization import serializable

from .scoring_rule import ScoringRule
from .exponential_score import _pairwise_diff


@serializable("bayesflow.scoring_rules", disable_module_check=True)
class AlphaExponentialScore(ScoringRule):
    r"""Alpha-exponential scoring rule for amortized Bayes factor estimation.

    Extends :class:`ExponentialScore` by weighting each pairwise term with a
    polynomial factor that penalises large log-odds:

    :math:`S(\{f_k\}, m; \alpha) = \sum_k \left(1 + (f_k - f_m)^2\right)^\alpha \exp(f_k - f_m)`

    For :math:`\alpha = 0` this reduces to :class:`ExponentialScore`.  Larger
    :math:`\alpha` provides additional down-weighting of extreme Bayes factors,
    which can improve gradient behaviour early in training.

    Parameters
    ----------
    alpha : float, optional
        Polynomial weight exponent (default: 0.5).  Must be non-negative.
    """

    NOT_TRANSFORMING_LIKE_VECTOR_WARNING = ("log_bayes_factors",)
    # Small-stddev init keeps initial log-odds near zero, preventing exp() overflow at the start of training.
    _head_kernel_initializer = keras.initializers.TruncatedNormal(mean=0.0, stddev=0.01)

    def __init__(self, alpha: float = 0.5, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.config = {"alpha": alpha}

    def get_head_shapes_from_target_shape(self, target_shape: Shape) -> dict[str, Shape]:
        target_shape = tuple(target_shape)
        return dict(log_bayes_factors=target_shape[1:-1] + (target_shape[-1] - 1,))

    def score(self, estimates: dict[str, Tensor], targets: Tensor, weights: Tensor = None) -> Tensor:
        """
        Computes the alpha-exponential Bayes factor score.

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
            (Optionally weighted) mean alpha-exponential score over the batch.
        """
        diff = _pairwise_diff(estimates["log_bayes_factors"], targets)
        mask = 1.0 - targets
        M = keras.ops.cast(keras.ops.shape(diff)[-1], dtype="float32")
        clip_max = 88.0 - keras.ops.log(keras.ops.maximum(M - 1.0, 1.0))
        # Compute weight * exp(diff/2) in log-domain to prevent joint overflow when weight is large
        log_term = self.alpha * keras.ops.log(1.0 + diff**2) + diff / 2.0
        scores = keras.ops.sum(
            mask * keras.ops.exp(keras.ops.minimum(keras.ops.maximum(log_term, -88.0), clip_max)), axis=-1
        )
        return weighted_mean(scores, weights)

    def get_config(self):
        return super().get_config() | self.config
