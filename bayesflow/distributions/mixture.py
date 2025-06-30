from collections.abc import Sequence

import numpy as np

import keras
from keras import ops

from bayesflow.types import Shape, Tensor
from bayesflow.utils.decorators import allow_batch_size
from bayesflow.utils.serialization import serializable, serialize
from bayesflow.distributions import Distribution


@serializable("bayesflow.distributions")
class Mixture(Distribution):
    """Utility class for a backend-agnostic mixture distributions."""

    def __init__(
        self,
        distributions: Sequence[Distribution],
        mixture_logits: Sequence[float] = None,
        trainable_mixture: bool = False,
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
        **kwargs
            Additional keyword arguments passed to the base `Distribution` class.

        Attributes
        ----------
        distributions : Sequence[Distribution]
            The list of component distributions.
        mixture_logits : Tensor
            Trainable or fixed logits representing the mixture weights.
        dim : int or None
            Dimensionality of the output samples; set when first sampling.
        """

        super().__init__(**kwargs)

        self.distributions = distributions

        if mixture_logits is None:
            self.mixture_logits = ops.ones(shape=len(distributions))
        else:
            self.mixture_logits = ops.convert_to_tensor(mixture_logits)

        self.trainable_mixture = trainable_mixture

        self.dim = None
        self._mixture_logits = None

    @allow_batch_size
    def sample(self, batch_shape: Shape) -> Tensor:
        """
        Draws samples from the mixture distribution by sampling a categorical index
        for each entry in `batch_shape` according to the softmax of `mixture_logits`,
        then draws from the corresponding component distribution.

        Parameters
        ----------
        batch_shape : Shape
            The desired sample batch shape (tuple of ints), not including the
            event dimension.

        Returns
        -------
        samples: Tensor
            A tensor of shape `batch_shape + (dim,)` containing samples drawn
            from the mixture.
        """
        # Will use numpy until keras adds support for N-D categorical sampling
        pvals = keras.ops.convert_to_numpy(keras.ops.softmax(self._mixture_logits))
        cat_samples = np.random.multinomial(n=1, pvals=pvals, size=batch_shape)
        cat_samples = cat_samples.argmax(axis=-1)

        # Prepare array to fill and dtype to infer
        samples = np.zeros(batch_shape + (self.dim,))
        dtype = None

        # Fill in array with vectorized sampling per component
        for i in range(len(self.distributions)):
            dist_mask = cat_samples == i
            dist_indices = np.where(dist_mask)
            num_dist_samples = np.sum(dist_mask)
            dist_samples = keras.ops.convert_to_numpy(self.distributions[i].sample(num_dist_samples))

            samples[dist_indices] = dist_samples

            dtype = dtype or keras.ops.dtype(dist_samples)

        # Convert to keras for compatibility
        samples = keras.ops.convert_to_tensor(samples, dtype=dtype)

        return samples

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
        }

        return base_config | serialize(config)
