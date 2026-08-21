import keras

from bayesflow.types import Shape, Tensor
from bayesflow.utils import weighted_mean
from bayesflow.utils.serialization import serializable

from .categorical_scoring_rule import CategoricalScoringRule


@serializable("bayesflow.scoring_rules", disable_module_check=True)
class PolynomialScore(CategoricalScoringRule):
    r""":math:`S(\hat p_{1\ldots C}, y; \alpha) = \tfrac{\alpha-1}{\alpha}\sum_c \hat p_c^\alpha - \hat p_y^{\alpha-1}`

    Implements the Tsallis proper scoring rule on the probability simplex,
    derived from the Savage representation with :math:`G(p) = \frac{1}{\alpha}\sum_k p_k^\alpha`:

    .. math::

        S(p, m; \alpha)
        = \frac{\alpha - 1}{\alpha}\sum_k p_k^\alpha - p_m^{\alpha - 1}

    where :math:`p = \mathrm{softmax}(\hat y)` and :math:`m` is the true model index.
    The unique minimizer of the expected score is the true posterior :math:`p_k^* = P(\mathcal{M}_k \mid x)`
    for any :math:`\alpha > 1`.

    For :math:`\alpha = 2` this is proportional to the :class:`BrierScore`
    (same gradient direction, same minimizer).  Larger :math:`\alpha` sharpens the
    penalty for wrong predictions.

    Parameters
    ----------
    alpha : float, optional
        Exponent (default: 2.0).  Must satisfy :math:`\alpha > 1`.
    """

    def __init__(self, alpha: float = 2.0, links=None, **kwargs):
        if alpha <= 1:
            raise ValueError(f"alpha must be greater than 1, got {alpha!r}.")
        super().__init__(links=links, **kwargs)
        self.alpha = alpha
        self.config = {"alpha": alpha}
        self.links = links or {"probs": "softmax"}

    def get_head_shapes_from_target_shape(self, target_shape: Shape) -> dict[str, Shape]:
        target_shape = tuple(target_shape)
        return dict(probs=target_shape[1:])

    def to_log_odds(self, rule_output: dict[str, Tensor]) -> Tensor:
        return keras.ops.log(rule_output["probs"])

    def score(self, estimates: dict[str, Tensor], targets: Tensor, weights: Tensor = None) -> Tensor:
        """
        Computes the Tsallis polynomial score from probabilities.

        Parameters
        ----------
        estimates : dict[str, Tensor]
            Must contain ``"probs"`` with shape
            ``(..., num_models)``.
        targets : Tensor
            One-hot encoded target labels of shape ``(..., num_models)``.
        weights : Tensor, optional
            Per-sample weights for a weighted mean.

        Returns
        -------
        Tensor
            (Optionally weighted) mean Tsallis polynomial score over the batch.
        """
        probs = estimates["probs"]
        scores = keras.ops.sum(
            (self.alpha - 1.0) / self.alpha * probs**self.alpha - targets * probs ** (self.alpha - 1.0),
            axis=-1,
        )
        return weighted_mean(scores, weights)

    def get_config(self):
        return super().get_config() | self.config
