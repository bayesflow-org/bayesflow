import keras

from bayesflow.types import Tensor
from bayesflow.utils.serialization import serializable

from .exponential_score import ExponentialScore


def _leaky_transform(x: Tensor, alpha: float, eps: float = 1e-8) -> Tensor:
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
    return x + x * keras.ops.power(keras.ops.abs(x) + eps, alpha - 1.0)


class _LeakyLink(keras.Layer):
    """Applies the leaky transform element-wise as a head link function."""

    def __init__(self, alpha: float, eps: float = 1e-8, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.eps = eps

    def call(self, x):
        return _leaky_transform(x, self.alpha, self.eps)

    def get_config(self):
        return super().get_config() | {"alpha": self.alpha, "eps": self.eps}


@serializable("bayesflow.scoring_rules", disable_module_check=True)
class LeakyExponentialScore(ExponentialScore):
    r"""Leaky exponential scoring rule for amortized Bayes factor estimation.

    **Recommended** Bayes factor scoring rule (Jeffrey & Wandelt 2024).

    Extends :class:`ExponentialScore` by applying the leaky parity-odd power
    (l-POP) transform :math:`J_\alpha(x) = x(1 + |x|^{\alpha-1})` as a head
    link function, so that the head output :math:`g = J_\alpha(f_\text{raw})`
    directly tracks :math:`\log K` at convergence.  The score is then plain
    :class:`ExponentialScore` in :math:`g`-space:

    .. math::

        S(\{g_k\}, m; \alpha) = \sum_{k \neq m}
            \exp\!\left(\tfrac{1}{2}(g_k - g_m)\right)

    The unique minimiser is :math:`g_k^* = \log K_{k,0}`.  Because the link
    is in the *head* (not inside :meth:`score`), partial convergence where
    :math:`g \approx c \cdot \log K` gives **linear** Bayes factor recovery at
    slope :math:`c` — not the parabolic artefact that results from applying
    :math:`J_\alpha` inside the loss.  Gradient amplification for large
    deviations is preserved via the chain rule.

    .. note::
        The leaky transform :math:`J_\alpha(x) = x(1 + |x|^{\alpha - 1})`
        keeps a unit-slope linear term near zero.  The paper's original
        :math:`\mathrm{lPOP}_\alpha(u) = \mathrm{sgn}(u)|u|^\alpha + \epsilon u`
        uses :math:`\epsilon \ll 1` instead; both are odd and strictly monotone.

    .. warning::
        Applying :math:`J_\alpha` to pairwise *differences*
        :math:`J_\alpha(f_k - f_m)` (instead of to individual values and then
        differencing) breaks properness for :math:`M > 2` because the nonlinear
        transform does not commute with subtraction.  The head link correctly
        applies :math:`J_\alpha` element-wise to each :math:`f_k` before the
        score computes pairwise differences.

    Parameters
    ----------
    alpha : float, optional
        Leaky exponent (default: 2.0).  Values :math:`> 1` give numerically
        stable gradients near zero; :math:`\alpha = 1` reduces to
        :class:`ExponentialScore`.
    """

    def __init__(self, alpha: float = 2.0, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.config = {"alpha": alpha}

    def get_link(self, key: str) -> keras.Layer:
        """Return the leaky transform as the head link for log_bayes_factors."""
        if key == "log_bayes_factors":
            return _LeakyLink(self.alpha)
        return super().get_link(key)

    def get_config(self):
        return super().get_config() | self.config
