import keras

from bayesflow.types import Shape, Tensor
from bayesflow.utils import weighted_mean
from bayesflow.utils.serialization import serializable

from .scoring_rule import ScoringRule
from .exponential_score import _pairwise_diff


@serializable("bayesflow.scoring_rules", disable_module_check=True)
class AlphaLogExponentialScore(ScoringRule):
    r"""Alpha-log-exponential scoring rule for amortized Bayes factor estimation.

    Interpolates between the :class:`LogisticScore` (:math:`\alpha \to 0^+`)
    and the :class:`ExponentialScore` (:math:`\alpha \to \infty`) via a
    scaled log-sum-exp over pairwise log-odds:

    :math:`S(\{f_k\}, m; \alpha) = \frac{1}{\alpha} \log \sum_k \exp\!\left(\alpha (f_k - f_m)\right)`

    For :math:`\alpha = 1` this equals
    :math:`\log \sum_k \exp(f_k - f_m)`, which is the log of the
    :class:`ExponentialScore` and equivalent to categorical cross-entropy.
    Smaller :math:`\alpha` softens the max-like behaviour, larger
    :math:`\alpha` sharpens it.

    Parameters
    ----------
    alpha : float, optional
        Temperature parameter (default: 1.0).  Must be positive.
    """

    NOT_TRANSFORMING_LIKE_VECTOR_WARNING = ("log_bayes_factors",)
    # Small-stddev init keeps initial log-odds near zero, preventing exp() overflow at the start of training.
    _head_kernel_initializer = keras.initializers.TruncatedNormal(mean=0.0, stddev=0.01)

    def __init__(self, alpha: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.config = {"alpha": alpha}

    def get_head_shapes_from_target_shape(self, target_shape: Shape) -> dict[str, Shape]:
        target_shape = tuple(target_shape)
        return dict(log_bayes_factors=target_shape[1:-1] + (target_shape[-1] - 1,))

    def score(self, estimates: dict[str, Tensor], targets: Tensor, weights: Tensor = None) -> Tensor:
        """
        Computes the alpha-log-exponential Bayes factor score.

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
            (Optionally weighted) mean alpha-log-exponential score over the batch.
        """
        diff = _pairwise_diff(estimates["log_bayes_factors"], targets)
        alpha_diff = self.alpha * diff
        # Numerically stable log-sum-exp: log(Σ exp(xᵢ)) = max + log(Σ exp(xᵢ - max))
        max_ad = keras.ops.max(alpha_diff, axis=-1, keepdims=True)
        log_sum_exp = max_ad[..., 0] + keras.ops.log(keras.ops.sum(keras.ops.exp(alpha_diff - max_ad), axis=-1))
        scores = log_sum_exp / self.alpha
        return weighted_mean(scores, weights)

    def get_config(self):
        return super().get_config() | self.config
