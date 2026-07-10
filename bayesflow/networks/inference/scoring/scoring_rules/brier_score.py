import keras

from bayesflow.types import Tensor
from bayesflow.utils import weighted_mean
from bayesflow.utils.serialization import serializable

from .polynomial_score import PolynomialScore


@serializable("bayesflow.scoring_rules", disable_module_check=True)
class BrierScore(PolynomialScore):
    r""":math:`S(\hat p_{1\ldots C}, y) = \sum_{c=1}^C (\hat p_c - y_c)^2`

    Minimized when the predicted probabilities exactly match the one-hot
    targets, i.e. when :math:`\mathrm{softmax}(\hat y)_m = 1` for the true
    model :math:`m`.

    .. note::
        Special case of :class:`PolynomialScore` with :math:`\alpha = 2`
        (same minimizer and gradient direction, different scale and offset).
        This class reports the true Brier score value rather than the Tsallis
        polynomial score.
    """

    def __init__(self, **kwargs):
        super().__init__(alpha=2.0, **kwargs)
        self.config = {}

    def score(self, estimates: dict[str, Tensor], targets: Tensor, weights: Tensor = None) -> Tensor:
        """
        Computes the Brier score from logits.

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
            (Optionally weighted) mean Brier score over the batch.
        """
        probs = keras.ops.softmax(estimates["logits"], axis=-1)
        scores = keras.ops.sum((probs - targets) ** 2, axis=-1)
        return weighted_mean(scores, weights)
