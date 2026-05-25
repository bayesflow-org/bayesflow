import keras

from bayesflow.types import Shape, Tensor
from bayesflow.utils import weighted_mean
from bayesflow.utils.serialization import serializable

from .scoring_rule import ScoringRule


@serializable("bayesflow.scoring_rules", disable_module_check=True)
class PolynomialScore(ScoringRule):
    r"""Polynomial (Tsallis) scoring rule for amortized model comparison.

    Implements the Tsallis proper scoring rule on the probability simplex,
    derived from the Savage representation with :math:`G(p) = \frac{1}{\alpha}\sum_k p_k^\alpha`:

    .. math::

        S(p, m; \alpha)
        = \frac{\alpha - 1}{\alpha}\sum_k p_k^\alpha - p_m^{\alpha - 1}

    where :math:`p = \mathrm{softmax}(\hat y)` and :math:`m` is the true model index.
    The unique minimiser of the expected score is the true posterior :math:`p_k^* = P(\mathcal{M}_k \mid x)`
    for any :math:`\alpha > 1`.

    For :math:`\alpha = 2` this is proportional to the :class:`BrierScore`
    (same gradient direction, same minimiser).  Larger :math:`\alpha` sharpens the
    penalty for confidently wrong predictions.

    Parameters
    ----------
    alpha : float, optional
        Exponent (default: 2.0).  Must satisfy :math:`\alpha > 1`.
    """

    def __init__(self, alpha: float = 2.0, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.config = {"alpha": alpha}

    def get_head_shapes_from_target_shape(self, target_shape: Shape) -> dict[str, Shape]:
        target_shape = tuple(target_shape)
        return dict(logits=target_shape[1:])

    def score(self, estimates: dict[str, Tensor], targets: Tensor, weights: Tensor = None) -> Tensor:
        """
        Computes the Tsallis polynomial score from logits.

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
            (Optionally weighted) mean Tsallis polynomial score over the batch.
        """
        targets = keras.ops.convert_to_tensor(targets)
        probs = keras.ops.softmax(estimates["logits"], axis=-1)
        scores = keras.ops.sum(
            (self.alpha - 1.0) / self.alpha * probs**self.alpha - targets * probs ** (self.alpha - 1.0),
            axis=-1,
        )
        return weighted_mean(scores, weights)

    def get_config(self):
        return super().get_config() | self.config
