import keras

from bayesflow.types import Shape, Tensor
from bayesflow.utils import weighted_mean
from bayesflow.utils.serialization import serializable

from .scoring_rule import ScoringRule


def _pairwise_diff(f: Tensor, targets: Tensor) -> Tensor:
    """Prepend f_0=0 and compute f_k - f_m for all k, where m is the true model."""
    targets = keras.ops.convert_to_tensor(targets)
    zeros = keras.ops.zeros_like(f[..., :1])
    f_full = keras.ops.concatenate([zeros, f], axis=-1)  # (..., M)
    m = keras.ops.cast(keras.ops.argmax(targets, axis=-1), dtype="int32")
    m_idx = keras.ops.expand_dims(m, axis=-1)  # (..., 1)
    f_m = keras.ops.take_along_axis(f_full, m_idx, axis=-1)  # (..., 1)
    return f_full - f_m  # (..., M), broadcast


@serializable("bayesflow.scoring_rules", disable_module_check=True)
class ScaledExponentialScore(ScoringRule):
    r"""Scaled exponential scoring rule for amortized Bayes factor estimation.

    Generalises :class:`ExponentialScore` by scaling the exponent by
    :math:`\alpha`:

    .. math::

        S(\{f_k\}, m; \alpha)
        = \sum_{k \neq m} \exp\!\left(\frac{\alpha}{2}(f_k(x) - f_m(x))\right)

    The unique minimiser of the expected loss is

    .. math::

        f_k^*(x) = \frac{1}{\alpha} \log K_{k,0}(x),

    so the network output must be multiplied by :math:`\alpha` to recover the
    true log-Bayes factor.  Setting :math:`\alpha = 1` recovers
    :class:`ExponentialScore` exactly.

    Parameters
    ----------
    alpha : float, optional
        Exponent scale (default: 1.0).  Must be positive.
    """

    NOT_TRANSFORMING_LIKE_VECTOR_WARNING = ("log_bayes_factors",)
    # Small-stddev init keeps initial log-odds near zero, preventing exp() overflow at the start of training.
    _head_kernel_initializer = keras.initializers.TruncatedNormal(mean=0.0, stddev=0.01)

    def __init__(self, alpha: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.config = {"alpha": alpha}

    def get_head_shapes_from_target_shape(self, target_shape: Shape) -> dict[str, Shape]:
        target_shape = tuple(target_shape)
        return dict(log_bayes_factors=target_shape[1:-1] + (target_shape[-1] - 1,))

    def score(self, estimates: dict[str, Tensor], targets: Tensor, weights: Tensor = None) -> Tensor:
        """
        Computes the scaled exponential Bayes factor score.

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
            (Optionally weighted) mean scaled exponential score over the batch.
        """
        targets = keras.ops.convert_to_tensor(targets)
        diff = _pairwise_diff(estimates["log_bayes_factors"], targets)
        mask = 1.0 - targets
        M = keras.ops.cast(keras.ops.shape(diff)[-1], dtype="float32")
        clip_max = 88.0 - keras.ops.log(keras.ops.maximum(M - 1.0, 1.0))
        alpha_half_diff = self.alpha * diff / 2.0
        scores = keras.ops.sum(
            mask * keras.ops.exp(keras.ops.minimum(keras.ops.maximum(alpha_half_diff, -88.0), clip_max)),
            axis=-1,
        )
        return weighted_mean(scores, weights)

    def to_bayes_factors(self, f: Tensor) -> Tensor:
        """Scale network outputs by alpha to recover log Bayes factors."""
        return f * self.alpha

    def get_config(self):
        return super().get_config() | self.config
