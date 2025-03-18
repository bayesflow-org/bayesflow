from keras.saving import register_keras_serializable as serializable

from .normed_difference_score import NormedDifferenceScore


@serializable(package="bayesflow.scores")
class MeanScore(NormedDifferenceScore):
    r""":math:`S(\hat \theta, \theta) = | \hat \theta - \theta |^2`

    Scores a predicted mean with the squared error score.
    """

    def __init__(self, **kwargs):
        super().__init__(k=2, **kwargs)
        self.config = {}
