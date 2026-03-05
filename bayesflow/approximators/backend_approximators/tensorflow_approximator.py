from typing import Any

import keras
import tensorflow as tf

from bayesflow.utils import filter_kwargs


class TensorFlowApproximator(keras.Model):
    """Backend-specific base for the TensorFlow/Keras training loop.

    Implements :meth:`train_step` and :meth:`test_step` using
    ``tf.GradientTape`` for automatic differentiation.  Metric
    tracking is handled by :meth:`_update_metrics`, which lazily
    creates :class:`keras.metrics.Mean` trackers as new metric names
    appear.

    Subclasses must implement
    :meth:`compute_metrics` and :meth:`_batch_size_from_data`
    (see :class:`BackendApproximator` for details).
    """

    # noinspection PyMethodOverriding
    def compute_metrics(self, *args, **kwargs) -> dict[str, tf.Tensor]:
        raise NotImplementedError

    def test_step(self, data: dict[str, Any]) -> dict[str, tf.Tensor]:
        """Perform a single validation step.

        Parameters
        ----------
        data : dict[str, Any]
            Batch data dict produced by the dataset.

        Returns
        -------
        dict[str, tf.Tensor]
            Computed validation metrics.
        """
        kwargs = filter_kwargs(data | {"stage": "validation"}, self.compute_metrics)
        metrics = self.compute_metrics(**kwargs)
        self._update_metrics(metrics, self._batch_size_from_data(data))
        return metrics

    def train_step(self, data: dict[str, Any]) -> dict[str, tf.Tensor]:
        """Perform a single training step with gradient update.

        Computes the forward pass inside a ``tf.GradientTape``, then
        applies gradients via the optimizer.

        Parameters
        ----------
        data : dict[str, Any]
            Batch data dict produced by the dataset.

        Returns
        -------
        dict[str, tf.Tensor]
            Computed training metrics.
        """
        with tf.GradientTape() as tape:
            kwargs = filter_kwargs(data | {"stage": "training"}, self.compute_metrics)
            metrics = self.compute_metrics(**kwargs)

        loss = metrics["loss"]

        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        self._update_metrics(metrics, self._batch_size_from_data(data))
        return metrics

    def _update_metrics(self, metrics: dict[str, Any], sample_weight: tf.Tensor = None):
        """Update internal Keras metric trackers with the given values.

        New metric names are lazily registered as
        :class:`keras.metrics.Mean` instances.

        Parameters
        ----------
        metrics : dict[str, Any]
            Computed metric values for the current batch.
        sample_weight : tf.Tensor, optional
            Optional sample weights applied during the update.
        """
        for name, value in metrics.items():
            try:
                metric_index = self.metrics_names.index(name)
                self.metrics[metric_index].update_state(value, sample_weight=sample_weight)
            except ValueError:
                self._metrics.append(keras.metrics.Mean(name=name))
                self._metrics[-1].update_state(value, sample_weight=sample_weight)

    # noinspection PyMethodOverriding
    def _batch_size_from_data(self, data: Any) -> int:
        raise NotImplementedError(
            "Correct calculation of the metrics requires obtaining the batch size from the supplied data "
            "for proper weighting of metrics for batches with different sizes. Please implement the "
            "_batch_size_from_data method for your approximator. For a given batch of data, it should "
            "return the corresponding batch size."
        )
