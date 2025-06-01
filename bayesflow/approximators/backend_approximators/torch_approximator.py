import keras
import torch

from bayesflow.utils import filter_kwargs


class TorchApproximator(keras.Model):
    """
    Base class for approximators using PyTorch and Keras-compatible training loops.

    This class defines a generic structure for metric computation, training, and evaluation
    using PyTorch backends. Subclasses must implement the `compute_metrics` method to return
    task-specific metrics, and the `_batch_size_from_data` method to extract batch size
    from the input data dictionary.

    Notes
    -----
    Subclasses must implement:
        - compute_metrics(self, *args, **kwargs) -> dict[str, torch.Tensor]
        - _batch_size_from_data(self, data: dict[str, any]) -> int
    """

    # noinspection PyMethodOverriding
    def compute_metrics(self, *args, **kwargs) -> dict[str, torch.Tensor]:
        """
        Compute and return a dictionary of metrics for the current batch.

        This method must be implemented by subclasses to compute relevant metrics
        (e.g., loss, accuracy) using PyTorch tensors. Inputs are dynamically filtered
        based on the metric function signature.

        Parameters
        ----------
        *args : tuple
            Positional arguments passed to the metric computation function.
        **kwargs
            Keyword arguments passed to the metric computation function.

        Returns
        -------
        dict of str to torch.Tensor
            Dictionary of metric names and their corresponding torch tensors.
        """
        raise NotImplementedError

    def test_step(self, data: dict[str, any]) -> dict[str, torch.Tensor]:
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

    def train_step(self, data: dict[str, any]) -> dict[str, torch.Tensor]:
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
        with torch.enable_grad():
            kwargs = filter_kwargs(data | {"stage": "training"}, self.compute_metrics)
            metrics = self.compute_metrics(**kwargs)

        loss = metrics["loss"]

        # noinspection PyUnresolvedReferences
        self.zero_grad()
        loss.backward()

        trainable_weights = self.trainable_weights[:]
        gradients = [v.value.grad for v in trainable_weights]

        # Update weights
        with torch.no_grad():
            self.optimizer.apply(gradients, trainable_weights)

        self._update_metrics(metrics, self._batch_size_from_data(data))
        return metrics

    def _update_metrics(self, metrics: dict[str, any], sample_weight: torch.Tensor = None):
        """
        Updates internal Keras metric trackers using provided metric values.

        Adds new metrics to the tracker list as `keras.metrics.Mean` instances if
        not already present.

        Parameters
        ----------
        metrics : dict of str to any
            Dictionary of computed metrics for the current batch.
        sample_weight : torch.Tensor, optional
            Sample weights used during metric updates.
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
