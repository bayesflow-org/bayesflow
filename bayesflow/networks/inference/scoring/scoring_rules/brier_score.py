import keras

from bayesflow.types import Shape, Tensor
from bayesflow.utils import weighted_mean
from bayesflow.utils.serialization import serializable

from .categorical_scoring_rule import CategoricalScoringRule


@serializable("bayesflow.scoring_rules", disable_module_check=True)
class BrierScore(CategoricalScoringRule):
    r""":math:`S(\hat p_{1\ldots C}, y) = \sum_{c=1}^C (\hat p_c - y_c)^2`

    Minimised when the predicted probabilities exactly match the one-hot
    targets, i.e. when :math:`\mathrm{softmax}(\hat y)_m = 1` for the true
    model :math:`m`.

    .. note::
        Proportional to :class:`PolynomialScore` with :math:`\alpha = 2`
        (same minimiser and gradient direction, different scale).
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.config = {}

    def get_head_shapes_from_target_shape(self, target_shape: Shape) -> dict[str, Shape]:
        target_shape = tuple(target_shape)
        return dict(logits=target_shape[1:])

    def score(self, estimates: dict[str, Tensor], targets: Tensor, weights: Tensor = None) -> Tensor:
        """
        Computes the Brier score from logits.

        Parameters
        ----------
        estimates : dict[str, Tensor]
            Must contain ``"logits"`` — raw (unnormalised) scores of shape
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
        targets = keras.ops.convert_to_tensor(targets)
        probs = keras.ops.softmax(estimates["logits"], axis=-1)
        scores = keras.ops.sum((probs - targets) ** 2, axis=-1)
        return weighted_mean(scores, weights)

    def get_config(self):
        return super().get_config() | self.config
