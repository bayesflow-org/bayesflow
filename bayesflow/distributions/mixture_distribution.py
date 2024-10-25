import keras
from keras import ops
from keras.saving import register_keras_serializable as serializable

from bayesflow.types import Shape, Tensor
from .distribution import Distribution


@serializable(package="bayesflow.distributions")
class MixtureDistribution(Distribution):
    """Utility class for a backend-agnostic mixture distributions."""

    def __init__(
        self,
        distributions: list[Distribution],
        mixture_logits: list[float] = None,
        trainable_mixture: bool = False,
        **kwargs,
    ):
        """TODO"""

        super().__init__(**kwargs)

        self.distributions = distributions

        if mixture_logits is None:
            mixture_logits = keras.ops.ones(shape=len(distributions))

        self.mixture_logits = self.add_weight(
            shape=(len(distributions),),
            initializer=keras.initializers.Constant(value=mixture_logits),
            dtype="float32",
            trainable=trainable_mixture,
        )

    def sample(self, batch_shape: Shape) -> Tensor:
        # TODO - Implement efficiently
        raise NotImplementedError

    def log_prob(self, x: Tensor, *, normalize: bool = True) -> Tensor:
        log_prob = [distribution.log_prob(x, normalize=normalize) for distribution in self.distributions]
        log_prob = ops.stack(log_prob, axis=-1)
        log_prob = ops.logsumexp(log_prob + ops.log_softmax(self.mixture_logits), axis=-1)
        return log_prob

    def build(self, input_shape: Shape) -> None:
        for distribution in self.distributions:
            distribution.build(input_shape)
