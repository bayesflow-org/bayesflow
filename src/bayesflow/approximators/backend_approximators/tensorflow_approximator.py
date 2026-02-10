import keras
import tensorflow as tf

from bayesflow.utils import filter_kwargs


class TensorFlowApproximator(keras.Model):
    """
    Base class for approximators using TensorFlow and Keras training logic.

    This class supports training and evaluation loops using TensorFlow backends.
    Subclasses are responsible for implementing the `compute_metrics` method and
    `_batch_size_from_data`, which extracts batch size information from data inputs.

    Notes
    -----
    Subclasses must implement:
        - compute_metrics(self, *args, **kwargs) -> dict[str, tf.Tensor]
        - _batch_size_from_data(self, data: dict[str, any]) -> int
    """

    # noinspection PyMethodOverriding
    def compute_metrics(self, *args, **kwargs) -> dict[str, tf.Tensor]:
        """
        Compute and return a dictionary of metrics for the current batch.

        This method is expected to be implemented by each subclass to compute task-specific
        metrics (e.g., loss, accuracy). The arguments are dynamically filtered based on the
        architecture's metric signature.

        Parameters
        ----------
        *args : tuple
            Positional arguments passed to the metric computation function.
        **kwargs
            Keyword arguments passed to the metric computation function.

        Returns
        -------
        dict of str to tf.Tensor
            Dictionary containing named metric values as TensorFlow tensors.
        """
        raise NotImplementedError

    def test_step(self, data: dict[str, any]) -> dict[str, tf.Tensor]:
        """
        Performs a single validation step.

        Filters relevant keyword arguments for metric computation and updates internal
        metric trackers using the validation data.

        Parameters
        ----------
        data : dict of str to any
            Input dictionary containing model inputs and possibly additional information
            such as sample_weight or mask.

        Returns
        -------
        dict of str to tf.Tensor
            Dictionary of computed validation metrics.
        """
        kwargs = filter_kwargs(data | {"stage": "validation"}, self.compute_metrics)
        metrics = self.compute_metrics(**kwargs)
        self._update_metrics(metrics, self._batch_size_from_data(data))
        return metrics

    def train_step(self, data: dict[str, any]) -> dict[str, tf.Tensor]:
        """
        Performs a single training step with gradient update.

        Computes gradients of the loss with respect to the trainable variables, applies
        the update, and updates internal metric trackers.

        Parameters
        ----------
        data : dict of str to any
            Input dictionary containing model inputs and training targets.

        Returns
        -------
        dict of str to tf.Tensor
            Dictionary of computed training metrics.
        """
        with tf.GradientTape() as tape:
            kwargs = filter_kwargs(data | {"stage": "training"}, self.compute_metrics)
            metrics = self.compute_metrics(**kwargs)

        loss = metrics["loss"]

        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        self._update_metrics(metrics, self._batch_size_from_data(data))
        return metrics

    def _update_metrics(self, metrics: dict[str, any], sample_weight: tf.Tensor = None):
        """
        Updates internal Keras metric objects with the given values.

        If a new metric name is encountered, it is added as a new `keras.metrics.Mean` instance.

        Parameters
        ----------
        metrics : dict of str to any
            Dictionary of computed metric values to update.
        sample_weight : tf.Tensor, optional
            Sample weights to apply during metric update.
        """
        for name, value in metrics.items():
            try:
                metric_index = self.metrics_names.index(name)
                self.metrics[metric_index].update_state(value, sample_weight=sample_weight)
            except ValueError:
                self._metrics.append(keras.metrics.Mean(name=name))
                self._metrics[-1].update_state(value, sample_weight=sample_weight)

    # noinspection PyMethodOverriding
    def _batch_size_from_data(self, data: any) -> int:
        """Obtain the batch size from a batch of data.

        To properly weigh the metrics for batches of different sizes, the batch size of a given batch of data is
        required. As the data structure differs between approximators, each concrete approximator has to specify
        this method.

        Parameters
        ----------
        data :
            The data that are passed to `compute_metrics` as keyword arguments.

        Returns
        -------
        batch_size : int
            The batch size of the given data.
        """
        raise NotImplementedError(
            "Correct calculation of the metrics requires obtaining the batch size from the supplied data "
            "for proper weighting of metrics for batches with different sizes. Please implement the "
            "_batch_size_from_data method for your approximator. For a given batch of data, it should "
            "return the corresponding batch size."
        )
