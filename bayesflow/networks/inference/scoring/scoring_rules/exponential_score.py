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


class _LeakyLink(keras.Layer):
    """Applies the leaky parity-odd power transform J_λ(x) = x(1 + |x|^{λ-1}) element-wise."""

    def __init__(self, power: float, eps: float = 1e-8, **kwargs):
        super().__init__(**kwargs)
        self.power = power
        self.eps = eps

    def call(self, x):
        return x + x * keras.ops.power(keras.ops.abs(x) + self.eps, self.power - 1.0)

    def get_config(self):
        return super().get_config() | {"power": self.power, "eps": self.eps}


@serializable("bayesflow.scoring_rules", disable_module_check=True)
class ExponentialScore(ScoringRule):
    r""":math:`S(\{f_k\}, m; \alpha) = \sum_{k \neq m} \exp\!\left(\tfrac{\alpha}{2}(f_k - f_m)\right)`

    The network learns :math:`M - 1` latent scores :math:`f_k` relative to
    reference model 0 (:math:`f_0 \equiv 0`). For the true model :math:`m`:

    .. math::

        S(\{f_k\}, m; \alpha) = \sum_{k \neq m}
            \exp\!\left(\frac{\alpha}{2}(f_k(x) - f_m(x))\right)

    The unique minimiser satisfies
    :math:`f_k^*(x) = \frac{1}{\alpha} \log K_{k,0}`,
    so network outputs are multiplied by :math:`\alpha` to recover log Bayes
    factors. Setting :math:`\alpha = 1` (default) gives the plain rule where
    the network directly estimates log Bayes factors.

    Optionally, a leaky parity-odd power (l-POP) head link
    :math:`J_\lambda(x) = x(1 + |x|^{\lambda - 1})` can be applied to each
    head output before the score computes pairwise differences. This improves
    numerical recovery of extreme log Bayes factors without affecting properness.

    Parameters
    ----------
    scale : float, optional
        Exponent scale (default: 1.0). Must be positive. The network output is
        multiplied by ``scale`` to recover log Bayes factors.
    leaky : float or None, optional
        Power for the leaky head link (default: None, i.e. identity link).
        When set, applies :math:`J_\lambda(x) = x(1 + |x|^{\lambda - 1})` as
        a head link. Recommended value when used: ``leaky=2.0``.

    Notes
    -----
    Special cases recoverable by parameter choice:

    - ``ExponentialScore()`` — plain exponential rule (:math:`\alpha=1`, no leaky link)
    - ``ExponentialScore(scale=scale)`` — scaled exponential rule
    - ``ExponentialScore(leaky=2.0)`` — leaky exponential rule (recommended BF rule)
    - ``ExponentialScore(scale=scale, leaky=power)`` — scaled + leaky (combined)
    """

    NOT_TRANSFORMING_LIKE_VECTOR_WARNING = ("log_bayes_factors",)
    # Small-stddev init keeps initial log-odds near zero, preventing exp() overflow at the start of training.
    _head_kernel_initializer = keras.initializers.TruncatedNormal(mean=0.0, stddev=0.01)

    def __init__(self, scale: float = 1.0, leaky: float | None = None, **kwargs):
        super().__init__(**kwargs)
        self.scale = scale
        self.leaky = leaky
        self.config = {"scale": scale, "leaky": leaky}

    def get_head_shapes_from_target_shape(self, target_shape: Shape) -> dict[str, Shape]:
        target_shape = tuple(target_shape)
        return dict(log_bayes_factors=target_shape[1:-1] + (target_shape[-1] - 1,))

    def score(self, estimates: dict[str, Tensor], targets: Tensor, weights: Tensor = None) -> Tensor:
        """
        Computes the exponential Bayes factor score.

        Parameters
        ----------
        estimates : dict[str, Tensor]
            Must contain ``"log_bayes_factors"`` of shape ``(..., M-1)`` — latent scores
            :math:`f_k` for models :math:`k = 1, \\ldots, M-1` relative to reference model 0.
        targets : Tensor
            One-hot encoded true model labels of shape ``(..., M)``.
        weights : Tensor, optional
            Per-sample weights for a weighted mean.

        Returns
        -------
        Tensor
            (Optionally weighted) mean exponential score over the batch.
        """
        targets = keras.ops.convert_to_tensor(targets)
        diff = _pairwise_diff(estimates["log_bayes_factors"], targets)
        mask = 1.0 - targets
        M = keras.ops.cast(keras.ops.shape(diff)[-1], dtype="float32")
        clip_max = 88.0 - keras.ops.log(keras.ops.maximum(M - 1.0, 1.0))
        alpha_half_diff = self.scale * diff / 2.0
        scores = keras.ops.sum(
            mask * keras.ops.exp(keras.ops.minimum(keras.ops.maximum(alpha_half_diff, -88.0), clip_max)),
            axis=-1,
        )
        return weighted_mean(scores, weights)

    def get_link(self, key: str) -> keras.Layer:
        if key == "log_bayes_factors" and self.leaky is not None:
            return _LeakyLink(power=self.leaky)
        return super().get_link(key)

    def to_bayes_factors(self, f: Tensor) -> Tensor:
        """Scale network outputs by scale to recover log Bayes factors."""
        return f * self.scale

    def get_config(self):
        return super().get_config() | self.config
