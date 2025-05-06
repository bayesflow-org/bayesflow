import keras

from bayesflow.metrics.functional import maximum_mean_discrepancy
from bayesflow.types import Tensor
from bayesflow.utils import layer_kwargs, find_distribution
from bayesflow.utils.decorators import sanitize_input_shape
from bayesflow.utils.serialization import deserialize


class SummaryNetwork(keras.Layer):
    def __init__(self, base_distribution: str = None, **kwargs):
        super().__init__(**layer_kwargs(kwargs))
        self.base_distribution = find_distribution(base_distribution)

    @sanitize_input_shape
    def build(self, input_shape):
        x = keras.ops.zeros(input_shape)
        z = self.call(x)

        if self.base_distribution is not None:
            self.base_distribution.build(keras.ops.shape(z))

    @sanitize_input_shape
    def compute_output_shape(self, input_shape):
        return keras.ops.shape(self.call(keras.ops.zeros(input_shape)))

    def call(self, x: Tensor, **kwargs) -> Tensor:
        """
        :param x: Tensor of shape (batch_size, set_size, input_dim)

        :param kwargs: Additional keyword arguments.

        :return: Tensor of shape (batch_size, output_dim)
        """
        raise NotImplementedError

    def compute_metrics(self, x: Tensor, stage: str = "training", **kwargs) -> dict[str, Tensor]:
        outputs = self(x, training=stage == "training")

        metrics = {"outputs": outputs}

        if self.base_distribution is not None:
            samples = self.base_distribution.sample((keras.ops.shape(x)[0],))
            mmd = maximum_mean_discrepancy(outputs, samples)
            metrics["loss"] = keras.ops.mean(mmd)

            if stage != "training":
                # compute sample-based validation metrics
                for metric in self.metrics:
                    metrics[metric.name] = metric(outputs, samples)

        return metrics

    @classmethod
    def from_config(cls, config, custom_objects=None):
        return cls(**deserialize(config, custom_objects=custom_objects))
