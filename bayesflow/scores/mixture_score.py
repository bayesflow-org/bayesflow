import keras

from bayesflow.types import Shape, Tensor
from bayesflow.utils.serialization import serializable, serialize, deserialize

from .parametric_distribution_score import ParametricDistributionScore


@serializable("bayesflow.scores")
class MixtureScore(ParametricDistributionScore):
    r""":math:`S(\hat p_{\phi_{1\ldots K},w_{1\ldots K}},\theta)=-\sum_{k=1}^{K} \log w_k+\log(\hat p_{\phi_k}(\theta))`

    Log-score of mixture of parametric distribution components.

    Parameters
    ----------
    components : dict[str, ParametricDistributionScore]
        Mixture components. Dict order defines component ordering.
    weight_head : str, optional
        Name of the mixture logits head. Defaults to ``"mixture_logits"``.
    temperature : float, optional
        Initial mixture temperature. Defaults to ``1.0``.
    **kwargs
        Passed to :py:class:`ParametricDistributionScore` and :py:class:`ScoringRule`.

    Notes
    -----
    The score exposes a flat set of estimation heads so that a
    :py:class:`ScoringRuleInferenceNetwork`` can build all required heads automatically.

    The exposed heads are:

    - ``weight_head``: mixture logits of shape ``(K,)``, where ``K = len(components)``
    - ``f"{c}__{h}"``: for each component ``c`` and component head ``h``

    Mixture weights are represented as logits for numerical stability

        log w = log_softmax(logits / temperature)

    where ``temperature`` is a non-trainable ``keras.Variable`` (default: 1.0)
    that can be updated externally with :py:meth:`set_temperature` (e.g. for scheduling).

    Examples
    --------
    >>> # A network representing a mixture density of three MVN distributions
    >>> inference_network = bf.networks.ScoringRuleInferenceNetwork(
            scores=dict(mix=bf.scores.MixtureScore(
                components=dict(
                    mvn1=bf.scores.MultivariateNormalScore(),
                    mvn2=bf.scores.MultivariateNormalScore(),
                    mvn3=bf.scores.MultivariateNormalScore(),
                ),
            ))
        )
    """

    def __init__(
        self,
        components: dict[str, ParametricDistributionScore],
        weight_head: str = "mixture_logits",
        temperature: float = 1.0,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if not isinstance(components, dict) or len(components) < 2:
            raise ValueError("`components` must be a dict with at least 2 ParametricDistributionScores.")

        for name, score in components.items():
            if not isinstance(score, ParametricDistributionScore):
                raise TypeError(f"Component '{name}' must be a ParametricDistributionScore, got {type(score)}.")

        self.components = components
        self.component_names = list(components.keys())
        self.K = len(self.component_names)

        self.weight_head = weight_head

        # Temperature is a non-trainable variable so it can be updated by external schedules.
        self.temperature = keras.Variable(temperature, trainable=False, dtype="float32", name="mixture_temperature")

        # Ensure mixture logits are not inverse-standardized like inference variables.
        # Components may define their own transformation types.
        self.TRANSFORMATION_TYPE = {self.weight_head: "identity"}
        for c, score in self.components.items():
            for h, ttype in getattr(score, "TRANSFORMATION_TYPE", {}).items():
                self.TRANSFORMATION_TYPE[f"{c}__{h}"] = ttype

        not_transforming = [self.weight_head]
        for c, score in self.components.items():
            for h in getattr(score, "NOT_TRANSFORMING_LIKE_VECTOR_WARNING", ()):
                not_transforming.append(f"{c}__{h}")
        self.NOT_TRANSFORMING_LIKE_VECTOR_WARNING = tuple(not_transforming)

        self.config = {
            "components": serialize(self.components),
            "weight_head": self.weight_head,
            "temperature": float(temperature),
        }

    def get_config(self):
        base = super().get_config()
        return base | self.config

    @classmethod
    def from_config(cls, config):
        cfg = config.copy()
        cfg["components"] = deserialize(cfg["components"])
        return cls(**cfg)

    def get_head_shapes_from_target_shape(self, target_shape: Shape) -> dict[str, Shape]:
        """Return the head shapes required to parameterize the mixture."""
        shapes: dict[str, Shape] = {self.weight_head: (self.K,)}
        for c, score in self.components.items():
            c_shapes = score.get_head_shapes_from_target_shape(target_shape)
            for h, s in c_shapes.items():
                shapes[f"{c}__{h}"] = s
        return shapes

    def get_head(self, key: str, output_shape: Shape) -> keras.Sequential:
        """
        Construct a head for the given key.

        - Mixture logits head is constructed via the base :py:class:`ScoringRule` logic.
        - Component heads are delegated to the respective component score, preserving its links/subnets.

        See also :py:meth:`ScoringRule.get_head`.
        """
        if key == self.weight_head:
            return super().get_head(key, output_shape)

        if "__" not in key:
            raise KeyError(f"Invalid head key '{key}'. Expected '<component>__<head_key>' or '{self.weight_head}'.")

        comp, head_key = key.split("__", 1)
        if comp not in self.components:
            raise KeyError(f"Unknown component '{comp}' in head key '{key}'.")
        return self.components[comp].get_head(head_key, output_shape)

    def set_temperature(self, value):
        self.temperature.assign(value)

    def _component_estimates(self, estimates: dict[str, Tensor], component: str) -> dict[str, Tensor]:
        """Extract a component's estimates dict by stripping the '<component>__' prefix."""
        prefix = f"{component}__"
        out: dict[str, Tensor] = {}
        for k, v in estimates.items():
            if k.startswith(prefix):
                out[k[len(prefix) :]] = v
        return out

    def log_prob(self, x: Tensor, **estimates: Tensor) -> Tensor:
        """
        Compute log p(x) under the mixture:

            log p(x) = logsumexp_k( log w_k + log p_k(x) )

        Parameters
        ----------
        x : Tensor
            Targets of shape (batch, ...event...).
        **estimates : dict[str, Tensor]
            Flat dict containing mixture logits and all component parameter heads.

        Returns
        -------
        Tensor
            Log-probabilities of shape (batch_size,).
        """
        logits = estimates[self.weight_head]
        log_w = keras.ops.log_softmax(logits / self.temperature, axis=-1)  # (batch_size, K)

        logps = []
        for c in self.component_names:
            c_est = self._component_estimates(estimates, c)
            lp = self.components[c].log_prob(x=x, **c_est)  # (batch_size,)
            logps.append(lp)
        logps = keras.ops.stack(logps, axis=-1)  # (batch_size, K)

        return keras.ops.logsumexp(log_w + logps, axis=-1)

    def sample(self, batch_shape: Shape, **estimates: Tensor) -> Tensor:
        """
        Draw samples from the mixture.

        Parameters
        ----------
        batch_shape : Shape
            A tuple (batch_size, num_samples).
        **estimates : dict[str, Tensor]
            Flat dict containing mixture logits and all component parameter heads.

        Returns
        -------
        Tensor
            Samples with shape (batch_size, num_samples, ...).
        """
        if len(batch_shape) != 2:
            raise ValueError("`batch_shape` must be a tuple (batch_size, num_samples).")

        batch_size, num_samples = batch_shape
        logits = estimates[self.weight_head]  # (batch_size, K) typically
        probs = keras.ops.softmax(logits / self.temperature, axis=-1)  # (batch_size, K)

        # Sample component indices via inverse-CDF on uniform draws
        u = keras.random.uniform((batch_size, num_samples, 1), dtype=probs.dtype)
        cdf = keras.ops.cumsum(probs, axis=-1)[:, None, :]  # (B, 1, K)
        cdf = keras.ops.broadcast_to(cdf, (batch_size, num_samples, self.K))  # (B, S, K)
        idx = keras.ops.argmax(keras.ops.cast(u <= cdf, "int32"), axis=-1)  # (B, S)

        # Sample from all components and select via one-hot mask
        comp_samples = []
        for c in self.component_names:
            c_est = self._component_estimates(estimates, c)
            s = self.components[c].sample((batch_size, num_samples), **c_est)
            comp_samples.append(s)

        stacked = keras.ops.stack(comp_samples, axis=2)  # (B, S, K, ...)
        mask = keras.ops.one_hot(idx, self.K, dtype=stacked.dtype)  # (B, S, K)

        # Expand mask to match stacked rank: (B, S, K, 1, 1, ...)
        while mask.ndim < stacked.ndim:
            mask = mask[..., None]

        return keras.ops.sum(stacked * mask, axis=2)
