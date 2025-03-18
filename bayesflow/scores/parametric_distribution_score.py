from bayesflow.types import Tensor

from .scoring_rule import ScoringRule


class ParametricDistributionScore(ScoringRule):
    r""":math:`S(\hat p_\phi, \theta; k) = \log(\hat p_\phi(\theta))`

    Base class for scoring a predicted parametric probability distribution with the log-score
    of the probability of the realized value.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def log_prob(self, *args, **kwargs):
        raise NotImplementedError

    def sample(self, *args, **kwargs):
        raise NotImplementedError

    def score(self, estimates: dict[str, Tensor], targets: Tensor, weights: Tensor = None) -> Tensor:
        """
        Computes the quantile-based scoring function.

        This function extracts the "value" tensor from the `estimates` dictionary and computes
        the pointwise difference between the estimates and the targets, expanding the target
        dimensions as necessary.

        The scoring function applies a quantile-based transformation to the difference, computing the
        mean score across a specified axis. The final score is then aggregated, optionally applying weights.

        Parameters
        ----------
        estimates : dict[str, Tensor]
            A dictionary containing tensors of estimated values. The "value" key must be present.
        targets : Tensor
            A tensor of true target values. The shape is adjusted to align with estimates.
        weights : Tensor, optional
            A tensor of weights corresponding to each estimate-target pair. If provided, it is used
            to compute a weighted aggregate score.

        Returns
        -------
        Tensor
            The aggregated quantile-based score, computed using the absolute pointwise difference
            transformed by the quantile adjustment, optionally weighted.
        """
        scores = -self.log_prob(x=targets, **estimates)
        score = self.aggregate(scores, weights)
        # multipy to mitigate instability due to relatively high values of parametric score
        return score * 0.01
