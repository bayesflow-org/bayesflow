from bayesflow.types import Tensor
from bayesflow.utils.serialization import serializable

from .polynomial_score import PolynomialScore


@serializable("bayesflow.scoring_rules", disable_module_check=True)
class BrierScore(PolynomialScore):
    r""":math:`S(\hat p_{1\ldots C}, y) = \sum_{c=1}^C (\hat p_c - y_c)^2`

    The (multi-class) Brier score, realised as the :class:`PolynomialScore`
    special case with :math:`\alpha = 2`. It is minimized in expectation by the
    true posterior model probabilities, i.e. when :math:`\mathrm{softmax}(\hat y)_m = 1`
    for the true model :math:`m`.

    .. note::
        For one-hot targets the Brier score is an exact affine map of the
        :math:`\alpha = 2` Tsallis polynomial score,
        :math:`S_{\mathrm{Brier}} = 2\,S_{\mathrm{poly}} + 1`. The heavy lifting is
        therefore inherited from :class:`PolynomialScore`; :meth:`score` only applies
        this transform so that the reported value is the true Brier score. (The scale
        and offset are immaterial for Bayes-risk minimization — same minimizer and
        gradient direction — but keep the metric interpretable.)
    """

    _SCALE = 2.0
    _OFFSET = 1.0
    """
    Affine map from the alpha=2 Tsallis polynomial score to the Brier score (special case).
    """

    def __init__(self, **kwargs):
        super().__init__(alpha=2.0, **kwargs)
        self.config = {}

    def score(self, estimates: dict[str, Tensor], targets: Tensor, weights: Tensor = None) -> Tensor:
        """
        Computes the (optionally weighted) mean Brier score from logits.

        Reuses :meth:`PolynomialScore.score` (the :math:`\\alpha = 2` Tsallis score)
        and rescales it to the classical Brier score. Because ``weighted_mean`` is
        affine-preserving, applying the transform to the aggregated score is
        equivalent to applying it per sample.

        Parameters
        ----------
        estimates : dict[str, Tensor]
            Must contain ``"probs"`` with shape
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
        return self._SCALE * super().score(estimates, targets, weights) + self._OFFSET
