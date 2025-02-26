import keras

from bayesflow.metrics.functional import maximum_mean_discrepancy
from bayesflow.types import Tensor
from bayesflow.utils import find_distribution, keras_kwargs
from bayesflow.utils.decorators import sanitize_input_shape
from bayesflow.utils.serialization import Serializable


class SummaryNetwork(Serializable, keras.Layer):
    def __init__(self, base_distribution: str = None, **kwargs):
        super().__init__(**keras_kwargs(kwargs))
        self.initialize_config()

        self.base_distribution = find_distribution(base_distribution)

    @sanitize_input_shape
    def build(self, input_shape):
        if self.base_distribution is not None:
            output_shape = keras.ops.shape(self.call(keras.ops.zeros(input_shape)))
            self.base_distribution.build(output_shape)

    def call(self, x: Tensor, **kwargs) -> Tensor:
        """
        :param x: Tensor of shape (batch_size, set_size, input_dim)

        :param kwargs: Additional keyword arguments.

        :return: Tensor of shape (batch_size, output_dim)
        """
        raise NotImplementedError

    def compute_metrics(self, x: Tensor, stage: str = "training") -> dict[str, Tensor]:
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
