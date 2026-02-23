import keras

from bayesflow.types import Shape, Tensor
from bayesflow.utils import weighted_mean
from bayesflow.utils.serialization import serializable

from .scoring_rule import ScoringRule


@serializable("bayesflow.scores")
class NormedDifferenceScore(ScoringRule):
    r""":math:`S(\hat \theta, \theta; k) = | \hat \theta - \theta |^k`

    Scores a point estimate with the k-norm of the error.
    """

    def __init__(self, k: int, **kwargs):
        super().__init__(**kwargs)

        #: Exponent to absolute difference
        self.k = k

        self.config = {"k": k}

    def get_head_shapes_from_target_shape(self, target_shape: Shape):
        # keras.saving.load_model sometimes passes target_shape as a list, so we force a conversion
        target_shape = tuple(target_shape)
        return dict(value=target_shape[1:])

    def score(self, estimates: dict[str, Tensor], targets: Tensor, weights: Tensor = None) -> Tensor:
        r"""
        Computes the scoring function based on the absolute difference between **estimates** and **targets**.

        :math:`S(\hat \theta, \theta; k) = | \hat \theta - \theta |^k`

        This function extracts the Tensor named ``"value"`` from the **estimates** dictionary and computes
        the element-wise absolute difference between the estimates and the true targets. The
        difference is then exponentiated by :py:attr:`k`. The final score is computed using the
        :py:func:`aggregate()` method, which optionally applies weighting.

        Parameters
        ----------
        estimates : dict[str, Tensor]
            A dictionary containing tensors of estimated values. The "value" key must be present.
        targets : Tensor
            A tensor of true target values.
        weights : Tensor, optional
            A tensor of weights corresponding to each estimate-target pair. If provided, it is used
            to compute a weighted aggregate score.

        Returns
        -------
        Tensor
            The aggregated score based on the element-wise absolute difference raised to the power
            of `self.k`, optionally weighted.
        """
        estimates = estimates["value"]
        scores = keras.ops.absolute(estimates - targets) ** self.k
        score = weighted_mean(scores, weights)
        return score

    def get_config(self):
        base_config = super().get_config()
        return base_config | self.config
