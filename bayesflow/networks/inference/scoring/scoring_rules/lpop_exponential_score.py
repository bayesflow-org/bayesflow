import keras

from bayesflow.types import Shape, Tensor
from bayesflow.utils import weighted_mean
from bayesflow.utils.serialization import serializable

from .scoring_rule import ScoringRule
from .exponential_score import _pairwise_diff


def _lpop(x: Tensor, alpha: float, eps: float = 1e-8) -> Tensor:
    r"""Leaky parity-odd power (l-POP) transform.

    :math:`J_\alpha(x) = x + x \, |x|^{\alpha - 1}`

    The transform is odd (:math:`J_\alpha(-x) = -J_\alpha(x)`), preserves
    sign, and behaves approximately linearly near zero when
    :math:`\alpha > 1`, avoiding the vanishing-gradient issue of the plain
    power transform :math:`\mathrm{sign}(x)|x|^\alpha`.

    Parameters
    ----------
    x : Tensor
        Input tensor.
    alpha : float
        Exponent.  Values :math:`> 1` are recommended for numerical stability.
    eps : float, optional
        Small constant added to :math:`|x|` before raising to the power
        :math:`\alpha - 1` to avoid division by zero when :math:`\alpha < 1`.
    """
    return x + x * keras.ops.power(keras.ops.abs(x), alpha - 1.0)


@serializable("bayesflow.scoring_rules", disable_module_check=True)
class LPOPExponentialScore(ScoringRule):
    r"""l-POP Exponential scoring rule for amortized Bayes factor estimation.

    **Recommended** Bayes factor scoring rule (Jeffrey & Wandelt 2024).

    Applies the leaky parity-odd power (l-POP) transform
    :math:`J_\alpha(x) = x(1 + |x|^{\alpha-1})` to each latent score
    :math:`f_k` individually, then takes pairwise differences before the
    exponential:

    .. math::

        S(\{f_k\}, m; \alpha) = \sum_{k \neq m}
            \exp\!\left(\tfrac{1}{2}\bigl(J_\alpha(f_k) - J_\alpha(f_m)\bigr)\right)

    Equivalently, defining :math:`g_k = J_\alpha(f_k)`, this is the
    :class:`ExponentialScore` on the transformed variables :math:`g_k`.  The
    unique minimiser is therefore :math:`g_k^* = \log K_{k,0}`, i.e.\
    :math:`f_k^* = J_\alpha^{-1}(\log K_{k,0})`, for **any** number of models
    :math:`M`.  The practical advantage over the plain exponential rule is that
    :math:`J_\alpha` with :math:`\alpha > 1` amplifies the gradient signal for
    large deviations, improving recovery across many orders of magnitude in
    :math:`K_{k,0}`.

    .. note::
        The code implements :math:`J_\alpha(x) = x(1 + |x|^{\alpha - 1})`,
        which keeps a unit-slope linear term near zero.  The paper's original
        :math:`\mathrm{lPOP}_\alpha(u) = \mathrm{sgn}(u)|u|^\alpha + \epsilon u`
        uses :math:`\epsilon \ll 1` instead; both are odd and strictly monotone.

    .. warning::
        Applying :math:`J_\alpha` to pairwise *differences*
        :math:`J_\alpha(f_k - f_m)` (instead of to individual values and then
        differencing) breaks properness for :math:`M > 2` because the nonlinear
        transform does not commute with subtraction.  The implementation
        correctly applies :math:`J_\alpha` element-wise to each :math:`f_k`
        before forming pairwise differences.

    Parameters
    ----------
    alpha : float, optional
        l-POP exponent (default: 2.0).  Values :math:`> 1` give numerically
        stable gradients near zero; :math:`\alpha = 1` reduces to
        :class:`ExponentialScore`.
    """

    NOT_TRANSFORMING_LIKE_VECTOR_WARNING = ("log_bayes_factors",)
    # Small-stddev init keeps initial log-odds near zero, preventing exp() overflow at the start of training.
    _head_kernel_initializer = keras.initializers.TruncatedNormal(mean=0.0, stddev=0.01)

    def __init__(self, alpha: float = 2.0, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.config = {"alpha": alpha}

    def get_head_shapes_from_target_shape(self, target_shape: Shape) -> dict[str, Shape]:
        target_shape = tuple(target_shape)
        return dict(log_bayes_factors=target_shape[1:-1] + (target_shape[-1] - 1,))

    def score(self, estimates: dict[str, Tensor], targets: Tensor, weights: Tensor = None) -> Tensor:
        """
        Computes the l-POP exponential Bayes factor score.

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
            (Optionally weighted) mean l-POP exponential score over the batch.
        """
        g = _lpop(estimates["log_bayes_factors"], self.alpha)
        diff = _pairwise_diff(g, targets)
        mask = 1.0 - targets
        transformed = diff / 2.0
        # Adjust per-term clip so sum of (M-1) terms stays within float32 range
        M = keras.ops.cast(keras.ops.shape(diff)[-1], dtype="float32")
        clip_max = 88.0 - keras.ops.log(keras.ops.maximum(M - 1.0, 1.0))
        scores = keras.ops.sum(
            mask * keras.ops.exp(keras.ops.minimum(keras.ops.maximum(transformed, -88.0), clip_max)), axis=-1
        )
        return weighted_mean(scores, weights)

    def to_bayes_factors(self, f: Tensor) -> Tensor:
        """Apply the l-POP transform to convert network outputs to log Bayes factors."""
        return _lpop(f, self.alpha)

    def get_config(self):
        return super().get_config() | self.config
