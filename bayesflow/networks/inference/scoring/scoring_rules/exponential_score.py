import numpy as np
import keras

from bayesflow.types import Shape, Tensor
from bayesflow.utils import weighted_mean
from bayesflow.utils.serialization import serializable

from bayesflow.utils.keras_utils import logits_relative_to_target
from .categorical_scoring_rule import CategoricalScoringRule

# Largest integer x such that exp(x) does not overflow in float32.
FLOAT32_EXP_MAX = np.floor(np.log(np.finfo(np.float32).max))


@serializable("bayesflow.scoring_rules", disable_module_check=True)
class ExponentialScore(CategoricalScoringRule):
    r""":math:`S(\{f_k\}, m) = \sum_{k \neq m} \exp\!\left(\tfrac{1}{2}(f_k - f_m)\right)`

    The network outputs logits :math:`f` whose softmax is the posterior over
    models; pairwise differences :math:`f_k - f_j` estimate log Bayes factors
    :math:`\log K_{k,j}`.

    A :class:`~bayesflow.links.Leaky` head link can be passed via ``links`` to
    improve numerical recovery of extreme log Bayes factors without affecting
    properness.

    Examples
    --------
    >>> ExponentialScore()

    With a leaky l-POP head link (recommended for extreme Bayes factors):

    >>> import bayesflow as bf
    >>> ExponentialScore(links={"logits": bf.links.Leaky(power=2.0)})
    """

    NOT_TRANSFORMING_LIKE_VECTOR_WARNING = ("logits",)
    # Small-stddev init keeps initial logits near zero for stable early training.
    _head_kernel_initializer = keras.initializers.TruncatedNormal(mean=0.0, stddev=0.01)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.config = {}

    def get_head_shapes_from_target_shape(self, target_shape: Shape) -> dict[str, Shape]:
        target_shape = tuple(target_shape)
        return dict(logits=target_shape[1:])

    def score(self, estimates: dict[str, Tensor], targets: Tensor, weights: Tensor = None) -> Tensor:
        """
        Computes the exponential Bayes factor score.

        Parameters
        ----------
        estimates : dict[str, Tensor]
            Must contain ``"logits"`` of shape ``(..., M)``.
        targets : Tensor
            One-hot encoded true model labels of shape ``(..., M)``.
        weights : Tensor, optional
            Per-sample weights for a weighted mean.

        Returns
        -------
        Tensor
            (Optionally weighted) mean exponential score over the batch.
        """
        diff = logits_relative_to_target(estimates["logits"], targets)
        mask = 1.0 - targets
        M = keras.ops.cast(keras.ops.shape(diff)[-1], dtype="float32")
        clip_max = FLOAT32_EXP_MAX - keras.ops.log(keras.ops.maximum(M - 1.0, 1.0))
        half_diff = 0.5 * diff
        scores = keras.ops.sum(
            mask * keras.ops.exp(keras.ops.minimum(keras.ops.maximum(half_diff, -FLOAT32_EXP_MAX), clip_max)),
            axis=-1,
        )
        return weighted_mean(scores, weights)

    def get_config(self):
        return super().get_config() | self.config
