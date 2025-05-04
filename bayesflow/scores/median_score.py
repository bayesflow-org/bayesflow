from bayesflow.utils.serialization import serializable
from .normed_difference_score import NormedDifferenceScore


@serializable("bayesflow.scores")
class MedianScore(NormedDifferenceScore):
    r""":math:`S(\hat \theta, \theta) = | \hat \theta - \theta |`

    Scores a predicted median with the absolute error score.
    """

    def __init__(self, **kwargs):
        super().__init__(k=1, **kwargs)
        self.config = {}
