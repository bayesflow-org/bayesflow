import numpy as np
import keras

from bayesflow.types import Shape, Tensor
from bayesflow.utils import weighted_mean
from bayesflow.utils.serialization import serializable

from .categorical_scoring_rule import CategoricalScoringRule
from bayesflow.utils.keras_utils import logits_relative_to_target

# Largest integer x such that exp(x) does not overflow in float32.
FLOAT32_EXP_MAX = np.floor(np.log(np.finfo(np.float32).max))


@serializable("bayesflow.scoring_rules", disable_module_check=True)
class LogisticScore(CategoricalScoringRule):
    r""":math:`S(\{f_k\}, m; \alpha) = \sum_{k \neq m} \left(1 + e^{(f_k - f_m)/(\alpha+1)}\right)^\alpha`
    for :math:`\alpha > 0`, and its :math:`\alpha \to 0` limit

    .. math::

        S(\{f_k\}, m) = \sum_{k \neq m} \log\!\left(1 + e^{f_k - f_m}\right)

    for :math:`\alpha = 0` (default, the plain logistic rule).

    The network outputs logits :math:`f` whose softmax is the posterior over models.

    Parameters
    ----------
    alpha : float, optional
        Power exponent (default: 0.0, the plain logistic rule). Must be
        non-negative.

    Examples
    --------
    Plain logistic rule:

    >>> LogisticScore()

    Power logistic rule:

    >>> LogisticScore(alpha=1.0)
    """

    NOT_TRANSFORMING_LIKE_VECTOR_WARNING = ("logits",)
    # Small-stddev init keeps initial logits near zero, preventing exp() overflow at the start of training.
    _head_kernel_initializer = keras.initializers.TruncatedNormal(mean=0.0, stddev=0.01)

    def __init__(self, alpha: float = 0.0, **kwargs):
        if alpha < 0:
            raise ValueError(f"alpha must be non-negative, got {alpha!r}.")
        super().__init__(**kwargs)
        self.alpha = alpha
        self.config = {"alpha": alpha}

    def get_head_shapes_from_target_shape(self, target_shape: Shape) -> dict[str, Shape]:
        target_shape = tuple(target_shape)
        return dict(logits=target_shape[1:])

    def score(self, estimates: dict[str, Tensor], targets: Tensor, weights: Tensor = None) -> Tensor:
        """
        Computes the (power) logistic Bayes factor score.

        Parameters
        ----------
        estimates : dict[str, Tensor]
            Must contain ``"logits"`` of shape ``(..., M)``.
        targets : Tensor
            One-hot encoded true model labels of shape ``(..., M)``.
        weights : Tensor, optional
            Per-sample weights for a weighted mean.

        Returns
        -------
        Tensor
            (Optionally weighted) mean (power) logistic score over the batch.
        """
        diff = logits_relative_to_target(estimates["logits"], targets)
        mask = 1.0 - targets
        log_terms = keras.ops.softplus(diff / (self.alpha + 1.0))
        if self.alpha == 0:
            scores = keras.ops.sum(mask * log_terms, axis=-1)
        else:
            M = keras.ops.cast(keras.ops.shape(diff)[-1], dtype="float32")
            clip_max = FLOAT32_EXP_MAX - keras.ops.log(keras.ops.maximum(M - 1.0, 1.0))
            scores = keras.ops.sum(
                mask * keras.ops.exp(keras.ops.minimum(self.alpha * log_terms, clip_max)),
                axis=-1,
            )
        return weighted_mean(scores, weights)

    def get_config(self):
        return super().get_config() | self.config
