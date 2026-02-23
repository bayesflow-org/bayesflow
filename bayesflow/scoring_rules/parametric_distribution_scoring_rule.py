from bayesflow.types import Tensor
from bayesflow.utils import weighted_mean
from bayesflow.utils.serialization import serializable

from .scoring_rule import ScoringRule


@serializable("bayesflow.scores")
class ParametricDistributionScore(ScoringRule):
    r""":math:`S(\hat p_\phi, \theta; k) = -\log(\hat p_\phi(\theta))`

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
        r"""
        Computes the log-score for a predicted parametric probability distribution given realized **targets**.

        :math:`S(\hat p_\phi, \theta; k) = -\log(\hat p_\phi(\theta))`
        """
        scores = -self.log_prob(x=targets, **estimates)
        score = weighted_mean(scores, weights)
        return score
