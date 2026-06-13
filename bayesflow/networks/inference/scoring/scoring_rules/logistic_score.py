import keras

from bayesflow.types import Shape, Tensor
from bayesflow.utils import weighted_mean
from bayesflow.utils.serialization import serializable

from .categorical_scoring_rule import CategoricalScoringRule
from bayesflow.utils.keras_utils import pairwise_diff


@serializable("bayesflow.scoring_rules", disable_module_check=True)
class LogisticScore(CategoricalScoringRule):
    r""":math:`S(\{f_k\}, m) = \sum_{k \neq m} \log\!\left(1 + e^{f_k - f_m}\right)`

    The unique minimizer is :math:`f_k^*(x) = \log K_{k,0}(x)`, so the
    network directly estimates log Bayes factors.
    """

    NOT_TRANSFORMING_LIKE_VECTOR_WARNING = ("log_bayes_factors",)
    # Small-stddev init keeps initial log-odds near zero, preventing exp() overflow at the start of training.
    _head_kernel_initializer = keras.initializers.TruncatedNormal(mean=0.0, stddev=0.01)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.config = {}

    def get_head_shapes_from_target_shape(self, target_shape: Shape) -> dict[str, Shape]:
        target_shape = tuple(target_shape)
        return dict(log_bayes_factors=target_shape[1:-1] + (target_shape[-1] - 1,))

    def score(self, estimates: dict[str, Tensor], targets: Tensor, weights: Tensor = None) -> Tensor:
        """
        Computes the logistic Bayes factor score.

        Parameters
        ----------
        estimates : dict[str, Tensor]
            Must contain ``"log_bayes_factors"`` of shape ``(..., M-1)``.
        targets : Tensor
            One-hot encoded true model labels of shape ``(..., M)``.
        weights : Tensor, optional
            Per-sample weights for a weighted mean.

        Returns
        -------
        Tensor
            (Optionally weighted) mean logistic score over the batch.
        """
        targets = keras.ops.convert_to_tensor(targets)
        diff = pairwise_diff(estimates["log_bayes_factors"], targets)
        mask = 1.0 - targets
        scores = keras.ops.sum(mask * keras.ops.softplus(diff), axis=-1)
        return weighted_mean(scores, weights)

    def get_config(self):
        return super().get_config() | self.config
