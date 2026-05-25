from bayesflow.utils.serialization import serializable

from .scaled_exponential_score import ScaledExponentialScore


@serializable("bayesflow.scoring_rules", disable_module_check=True)
class ExponentialScore(ScaledExponentialScore):
    r"""Exponential scoring rule for amortized Bayes factor estimation.

    The network learns :math:`M - 1` latent scores :math:`f_k` relative to
    reference model 0 (:math:`f_0 \equiv 0`).  For the true model :math:`m`:

    .. math::

        S(\{f_k\}, m) = \sum_{k \neq m} \exp\!\left(\tfrac{1}{2}(f_k - f_m)\right)

    The unique minimiser of the expected loss satisfies
    :math:`f_k^* = \log K_{k,0} = \log p(x \mid \mathcal{M}_k) - \log p(x \mid \mathcal{M}_0)`,
    so the reference model is always in the denominator.

    Special case of :class:`ScaledExponentialScore` with :math:`\alpha = 1`.
    """

    def __init__(self, **kwargs):
        super().__init__(alpha=1, **kwargs)
        self.config = {}
