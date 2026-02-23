from bayesflow.utils.serialization import serializable
from .normed_difference_scoring_rule import NormedDifferenceScoringRule


@serializable("bayesflow.scoring_rules")
class MedianScoringRule(NormedDifferenceScoringRule):
    r""":math:`S(\hat \theta, \theta) = | \hat \theta - \theta |`

    Scores a predicted median with the absolute error score.
    """

    def __init__(self, **kwargs):
        super().__init__(k=1, **kwargs)
        self.config = {}
