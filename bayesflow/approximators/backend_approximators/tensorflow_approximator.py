import keras
import tensorflow as tf

from bayesflow.utils import filter_kwargs


class TensorFlowApproximator(keras.Model):
    # noinspection PyMethodOverriding
    def compute_metrics(self, *args, **kwargs) -> dict[str, tf.Tensor]:
        # implemented by each respective architecture
        raise NotImplementedError

    def test_step(self, data: dict[str, any]) -> dict[str, tf.Tensor]:
        kwargs = filter_kwargs(data | {"stage": "validation"}, self.compute_metrics)
        metrics = self.compute_metrics(**kwargs)
        self._update_metrics(metrics)
        return metrics

    def train_step(self, data: dict[str, any]) -> dict[str, tf.Tensor]:
        with tf.GradientTape() as tape:
            kwargs = filter_kwargs(data | {"stage": "training"}, self.compute_metrics)
            metrics = self.compute_metrics(**kwargs)

        loss = metrics["loss"]

        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        self._update_metrics(metrics)
        return metrics

    def _update_metrics(self, metrics):
        for name, value in metrics.items():
            try:
                metric_index = self.metrics_names.index(name)
                self.metrics[metric_index].update_state(value)
            except ValueError:
                self._metrics.append(keras.metrics.Mean(name=name))
                self._metrics[-1].update_state(value)
