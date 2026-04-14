import math
from collections.abc import Sequence

import keras
from keras import ops

from bayesflow.distributions import Distribution
from bayesflow.types import Shape, Tensor
from bayesflow.utils.decorators import allow_batch_size
from bayesflow.utils.keras_utils import resolve_seed
from bayesflow.utils.serialization import serializable, serialize


@serializable("bayesflow.distributions")
class Mixture(Distribution):
    """Utility class for a backend-agnostic mixture distributions."""

    def __init__(
        self,
        distributions: Sequence[Distribution],
        mixture_logits: Sequence[float] | None = None,
        trainable_mixture: bool = False,
        seed_generator: keras.random.SeedGenerator | None = None,
        **kwargs,
    ):
        """
        Initializes a mixture of distributions as a latent distro.

        Parameters
        ----------
        distributions : Sequence[Distribution]
            A sequence of `Distribution` instances to form the mixture components.
        mixture_logits : Sequence[float], optional
            Initial unnormalized log‑weights for each component. If `None`, all
            components are assigned equal weight. Default is `None`.
        trainable_mixture : bool, optional
            Whether the mixture weights (`mixture_logits`) should be trainable.
            Default is `False`.
        seed_generator : keras.random.SeedGenerator, optional
            Seed generator for reproducible sampling. If ``None``, a new one is created.
        **kwargs
            Additional keyword arguments passed to the base `Distribution` class.
        """

        super().__init__(**kwargs)

        self.distributions = distributions

        if mixture_logits is None:
            self.mixture_logits = ops.ones(shape=len(distributions))
        else:
            self.mixture_logits = ops.convert_to_tensor(mixture_logits)

        self.trainable_mixture = trainable_mixture
        self.seed_generator = seed_generator or keras.random.SeedGenerator()

        self.dim = None
        self._mixture_logits = None

    @allow_batch_size
    def sample(self, batch_shape: Shape, seed: int | keras.random.SeedGenerator | None = None) -> Tensor:
        """
        Draws samples from the mixture distribution by sampling a categorical index
        for each entry in `batch_shape` according to `mixture_logits`,
        then draws from the corresponding component distribution.

        Parameters
        ----------
        batch_shape : Shape
            The desired sample batch shape (tuple of ints), not including the
            event dimension.
        seed : int, keras.random.SeedGenerator, or None, optional
            Seed for reproducible sampling. An integer is converted to a
            ``keras.random.SeedGenerator`` and shared across all random draws in
            the call. A ``SeedGenerator`` is passed through as-is, advancing its
            state with each use. If ``None`` (default), the instance seed
            generator is used.

        Returns
        -------
        Tensor
            Samples with shape ``batch_shape + (event_dim,)``.
        """
        sg = resolve_seed(seed) or self.seed_generator
        K = len(self.distributions)
        total = math.prod(batch_shape)

        # Sample component indices: (total,)
        logits_broadcast = keras.ops.broadcast_to(keras.ops.expand_dims(self._mixture_logits, 0), (total, K))
        cat_indices = keras.ops.squeeze(keras.random.categorical(logits_broadcast, num_samples=1, seed=sg), axis=-1)

        # Sample from all components and select via one-hot mask (avoids dynamic shapes)
        all_flat = keras.ops.stack(
            [keras.ops.reshape(dist.sample(batch_shape, seed=sg), (total, self.dim)) for dist in self.distributions]
        )
        all_flat = keras.ops.transpose(all_flat, (1, 0, 2))

        one_hot = keras.ops.cast(keras.ops.one_hot(cat_indices, K), all_flat.dtype)  # (total, K)
        selected = keras.ops.sum(all_flat * one_hot[..., None], axis=1)  # (total, dim)

        return keras.ops.reshape(selected, batch_shape + (self.dim,))

    def log_prob(self, samples: Tensor, *, normalize: bool = True) -> Tensor:
        """
        Compute the log probability of given samples under the mixture.

        For each input sample, computes the weighted log‑sum‑exp of the component
        log‑probabilities plus the mixture log‑weights.

        Parameters
        ----------
        samples : Tensor
            A tensor of samples with shape `batch_shape + (dim,)`.
        normalize : bool, optional
            If `True`, returns normalized log‑probabilities (i.e., includes the
            log normalization constant). Default is `True`.

        Returns
        -------
        Tensor
            A tensor of shape `batch_shape`, containing the log probability of
            each sample under the mixture distribution.
        """

        log_prob = [distribution.log_prob(samples, normalize=normalize) for distribution in self.distributions]
        log_prob = ops.stack(log_prob, axis=-1)
        log_prob = ops.logsumexp(log_prob + ops.log_softmax(self._mixture_logits), axis=-1)
        return log_prob

    def build(self, input_shape: Shape) -> None:
        if self.built:
            return

        self.dim = input_shape[-1]

        for distribution in self.distributions:
            distribution.build(input_shape)

        self._mixture_logits = self.add_weight(
            shape=(len(self.distributions),),
            initializer=keras.initializers.get(keras.ops.copy(self.mixture_logits)),
            dtype="float32",
            trainable=self.trainable_mixture,
        )

    def get_config(self):
        base_config = super().get_config()

        config = {
            "distributions": self.distributions,
            "mixture_logits": self.mixture_logits,
            "trainable_mixture": self.trainable_mixture,
            "seed_generator": self.seed_generator,
        }

        return base_config | serialize(config)
