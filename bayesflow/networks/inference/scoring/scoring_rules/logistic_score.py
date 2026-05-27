import keras

from bayesflow.types import Shape, Tensor
from bayesflow.utils import weighted_mean
from bayesflow.utils.serialization import serializable

from .scoring_rule import ScoringRule
from .exponential_score import _pairwise_diff


@serializable("bayesflow.scoring_rules", disable_module_check=True)
class LogisticScore(ScoringRule):
    r""":math:`S(\{f_k\}, m) = \sum_{k \neq m} \log\!\left(1 + e^{f_k - f_m}\right)`

    Two modes selected by the ``alpha`` parameter:

    **Log mode** (``alpha=None``, default):

    .. math::

        S(\{f_k\}, m) = \sum_{k \neq m} \log\!\left(1 + e^{f_k(x) - f_m(x)}\right)

    The unique minimiser is :math:`f_k^*(x) = \log K_{k,0}(x)`.

    **Power mode** (``alpha`` is a positive float):

    .. math::

        S(\{f_k\}, m; \alpha)
        = \sum_{k \neq m} \left(1 + e^{f_k(x) - f_m(x)}\right)^\alpha

    The unique minimiser is :math:`f_k^*(x) = \tfrac{1}{\alpha+1}\log K_{k,0}(x)`,
    so the network output is multiplied by :math:`\alpha + 1` to recover the true log Bayes factor.

    Parameters
    ----------
    alpha : float or None, optional
        Exponent for power mode. ``None`` (default) uses log mode (softplus loss).
        A positive float uses power mode (:math:`\exp(\alpha \cdot \mathrm{softplus})` loss).
    """

    NOT_TRANSFORMING_LIKE_VECTOR_WARNING = ("log_bayes_factors",)
    # Small-stddev init keeps initial log-odds near zero, preventing exp() overflow at the start of training.
    _head_kernel_initializer = keras.initializers.TruncatedNormal(mean=0.0, stddev=0.01)

    def __init__(self, alpha: float | None = None, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.config = {"alpha": alpha}

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
        targets = keras.ops.convert_to_tensor(targets)
        diff = _pairwise_diff(estimates["log_bayes_factors"], targets)
        mask = 1.0 - targets

        if self.alpha is None:
            scores = keras.ops.sum(mask * keras.ops.softplus(diff), axis=-1)
        else:
            M = keras.ops.cast(keras.ops.shape(diff)[-1], dtype="float32")
            clip_max = 88.0 - keras.ops.log(keras.ops.maximum(M - 1.0, 1.0))
            # (1 + exp(diff))^alpha = exp(alpha * softplus(diff)); only upper clip needed
            log_terms = self.alpha * keras.ops.softplus(diff)
            scores = keras.ops.sum(
                mask * keras.ops.exp(keras.ops.minimum(log_terms, clip_max)),
                axis=-1,
            )

        return weighted_mean(scores, weights)

    def to_bayes_factors(self, f: Tensor) -> Tensor:
        """Scale network outputs by (alpha + 1) to recover log Bayes factors (power mode only)."""
        if self.alpha is None:
            return f
        return f * (self.alpha + 1)

    def get_config(self):
        return super().get_config() | self.config
