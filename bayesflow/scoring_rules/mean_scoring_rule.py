from bayesflow.utils.serialization import serializable
from .normed_difference_scoring_rule import NormedDifferenceScoringRule


@serializable("bayesflow.scoring_rules")
class MeanScoringRule(NormedDifferenceScoringRule):
    r""":math:`S(\hat \theta, \theta) = | \hat \theta - \theta |^2`

    Scores a predicted mean with the squared error score.
    """

    def __init__(self, **kwargs):
        super().__init__(k=2, **kwargs)
        self.config = {}
