import keras

from bayesflow.types import Shape, Tensor
from bayesflow.utils import weighted_mean
from bayesflow.utils.serialization import serializable

from .scoring_rule import ScoringRule
from .exponential_score import _pairwise_diff


def _lpop(x: Tensor, alpha: float, eps: float = 1e-8) -> Tensor:
    r"""Leaky parity-odd power (l-POP) transform.

    :math:`J_\alpha(x) = x + x \, |x|^{\alpha - 1}`

    The transform is odd (:math:`J_\alpha(-x) = -J_\alpha(x)`), preserves
    sign, and behaves approximately linearly near zero when
    :math:`\alpha > 1`, avoiding the vanishing-gradient issue of the plain
    power transform :math:`\mathrm{sign}(x)|x|^\alpha`.

    Parameters
    ----------
    x : Tensor
        Input tensor.
    alpha : float
        Exponent.  Values :math:`> 1` are recommended for numerical stability.
    eps : float, optional
        Small constant added to :math:`|x|` before raising to the power
        :math:`\alpha - 1` to avoid division by zero when :math:`\alpha < 1`.
    """
    return x + x * keras.ops.power(keras.ops.abs(x) + eps, alpha - 1.0)


@serializable("bayesflow.scoring_rules", disable_module_check=True)
class LPOPExponentialScore(ScoringRule):
    r"""l-POP Exponential scoring rule for amortized Bayes factor estimation.

    **Recommended** Bayes factor scoring rule (Jeffrey & Wandelt 2024).

    Applies the leaky parity-odd power (l-POP) transform
    :math:`J_\alpha(x) = x + x|x|^{\alpha-1}` to each pairwise log-odds
    before the exponential:

    :math:`S(\{f_k\}, m; \alpha) = \sum_k \exp\!\left(J_\alpha(f_k - f_m)\right)`

    The l-POP transform is odd and linear near zero (:math:`\alpha > 1`),
    which robustifies training against extreme Bayes factors while retaining
    the properness of the underlying exponential rule.

    Parameters
    ----------
    alpha : float, optional
        l-POP exponent (default: 2.0).  Values :math:`> 1` give numerically
        stable gradients near zero; :math:`\alpha = 1` reduces to
        :class:`ExponentialScore` (up to a factor of 2).
    """

    NOT_TRANSFORMING_LIKE_VECTOR_WARNING = ("log_bayes_factors",)
    # Small-stddev init keeps initial log-odds near zero, preventing exp() overflow at the start of training.
    _head_kernel_initializer = keras.initializers.TruncatedNormal(mean=0.0, stddev=0.01)

    def __init__(self, alpha: float = 2.0, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.config = {"alpha": alpha}

    def get_head_shapes_from_target_shape(self, target_shape: Shape) -> dict[str, Shape]:
        target_shape = tuple(target_shape)
        return dict(log_bayes_factors=target_shape[1:-1] + (target_shape[-1] - 1,))

    def score(self, estimates: dict[str, Tensor], targets: Tensor, weights: Tensor = None) -> Tensor:
        """
        Computes the l-POP exponential Bayes factor score.

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
            (Optionally weighted) mean l-POP exponential score over the batch.
        """
        diff = _pairwise_diff(estimates["log_bayes_factors"], targets)
        mask = 1.0 - targets
        transformed = _lpop(diff, self.alpha) / 2.0
        # Adjust per-term clip so sum of (M-1) terms stays within float32 range
        M = keras.ops.cast(keras.ops.shape(diff)[-1], dtype="float32")
        clip_max = 88.0 - keras.ops.log(keras.ops.maximum(M - 1.0, 1.0))
        scores = keras.ops.sum(
            mask * keras.ops.exp(keras.ops.minimum(keras.ops.maximum(transformed, -88.0), clip_max)), axis=-1
        )
        return weighted_mean(scores, weights)

    def get_config(self):
        return super().get_config() | self.config
