import keras

from bayesflow.types import Shape, Tensor
from bayesflow.utils import weighted_mean
from bayesflow.utils.serialization import serializable

from .scoring_rule import ScoringRule


@serializable("bayesflow.scoring_rules", disable_module_check=True)
class PolynomialScore(ScoringRule):
    r"""Polynomial scoring rule for amortized model comparison.

    Scores predicted logits against one-hot encoded targets using the
    polynomial proper scoring rule:

    :math:`S(\hat y, y; \alpha) = \sum_k \left[ y_k (1 - p_k)^\alpha + (1 - y_k)\, p_k^\alpha \right]`

    where :math:`p = \mathrm{softmax}(\hat y)`.  For :math:`\alpha = 2` this
    is equivalent to the Brier / :class:`SquaredScore`.  Larger :math:`\alpha`
    places more weight on confident but wrong predictions.

    Parameters
    ----------
    alpha : float, optional
        Exponent controlling the sharpness of the penalty (default: 2.0).
        Must be positive.
    """

    def __init__(self, alpha: float = 2.0, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.config = {"alpha": alpha}

    def get_head_shapes_from_target_shape(self, target_shape: Shape) -> dict[str, Shape]:
        target_shape = tuple(target_shape)
        return dict(logits=target_shape[1:])

    def score(self, estimates: dict[str, Tensor], targets: Tensor, weights: Tensor = None) -> Tensor:
        """
        Computes the polynomial score from logits.

        Parameters
        ----------
        estimates : dict[str, Tensor]
            Must contain ``"logits"`` — raw (unnormalised) scores of shape
            ``(..., num_models)``.
        targets : Tensor
            One-hot encoded target labels of shape ``(..., num_models)``.
        weights : Tensor, optional
            Per-sample weights for a weighted mean.

        Returns
        -------
        Tensor
            (Optionally weighted) mean polynomial score over the batch.
        """
        probs = keras.ops.sigmoid(estimates["logits"])
        scores = keras.ops.sum(
            targets * (1.0 - probs) ** self.alpha + (1.0 - targets) * probs**self.alpha,
            axis=-1,
        )
        return weighted_mean(scores, weights)

    def get_config(self):
        return super().get_config() | self.config
