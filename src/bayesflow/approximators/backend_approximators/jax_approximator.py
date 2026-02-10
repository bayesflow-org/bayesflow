import jax
import keras

from bayesflow.utils import filter_kwargs


class JAXApproximator(keras.Model):
    """
    Base class for approximators using JAX and Keras' stateless training interface.

    This class enables stateless training and evaluation steps with JAX, supporting
    JAX-compatible gradient computation and variable updates through the `StatelessScope`.

    Notes
    -----
    Subclasses must implement:
        - compute_metrics(self, *args, **kwargs) -> dict[str, jax.Array]
        - _batch_size_from_data(self, data: dict[str, any]) -> int
    """

    # noinspection PyMethodOverriding
    def compute_metrics(self, *args, **kwargs) -> dict[str, jax.Array]:
        """
        Compute and return a dictionary of metrics for the current batch.

        This method is expected to be implemented by each subclass to compute
        task-specific metrics using JAX arrays. It is compatible with stateless
        execution and must be differentiable under JAX's `grad` system.

        Parameters
        ----------
        *args : tuple
            Positional arguments passed to the metric computation function.
        **kwargs
            Keyword arguments passed to the metric computation function.

        Returns
        -------
        dict of str to jax.Array
            Dictionary containing named metric values as JAX arrays.
        """
        raise NotImplementedError

    def stateless_compute_metrics(
        self,
        trainable_variables: any,
        non_trainable_variables: any,
        metrics_variables: any,
        data: dict[str, any],
        stage: str = "training",
    ) -> (jax.Array, tuple):
        """
        Stateless computation of metrics required for JAX autograd.

        This method performs a stateless forward pass using the given model
        variables and returns both the loss and auxiliary information for
        further updates.

        Things we do for specifically jax:

        1. Accept trainable variables as the first argument
            (can be at any position as indicated by the argnum parameter
             in autograd, but needs to be an explicit arg)
        2. Accept, potentially modify, and return other state variables
        3. Return just the loss tensor as the first value
        4. Return all other values in a tuple as the second value

        This ensures:

        1. The function is stateless
        2. The function can be differentiated with jax autograd

        Parameters
        ----------
        trainable_variables : Any
            Current values of the trainable weights.
        non_trainable_variables : Any
            Current values of non-trainable variables (e.g., batch norm statistics).
        metrics_variables : Any
            Current values of metric tracking variables.
        data : dict of str to any
            Input data dictionary passed to `compute_metrics`.
        stage : str, default="training"
            Whether the computation is for "training" or "validation".

        Returns
        -------
        loss : jax.Array
            Scalar loss tensor for gradient computation.
        aux : tuple
            Tuple containing:
                - metrics (dict of str to jax.Array)
                - updated non-trainable variables
                - updated metrics variables
        """
        state_mapping = []
        state_mapping.extend(zip(self.trainable_variables, trainable_variables))
        state_mapping.extend(zip(self.non_trainable_variables, non_trainable_variables))
        state_mapping.extend(zip(self.metrics_variables, metrics_variables))

        # perform a stateless call to compute_metrics
        with keras.StatelessScope(state_mapping) as scope:
            kwargs = filter_kwargs(data | {"stage": stage}, self.compute_metrics)
            metrics = self.compute_metrics(**kwargs)

        # update variables
        non_trainable_variables = [scope.get_current_value(v) for v in self.non_trainable_variables]
        metrics_variables = [scope.get_current_value(v) for v in self.metrics_variables]

        return metrics["loss"], (metrics, non_trainable_variables, metrics_variables)

    def stateless_test_step(self, state: tuple, data: dict[str, any]) -> (dict[str, jax.Array], tuple):
        """
        Stateless validation step compatible with JAX.

        Parameters
        ----------
        state : tuple
            Tuple of (trainable_variables, non_trainable_variables, metrics_variables).
        data : dict of str to any
            Input data for validation.

        Returns
        -------
        metrics : dict of str to jax.Array
            Dictionary of computed evaluation metrics.
        state : tuple
            Updated state tuple after evaluation.
        """
        trainable_variables, non_trainable_variables, metrics_variables = state

        loss, aux = self.stateless_compute_metrics(
            trainable_variables, non_trainable_variables, metrics_variables, data=data, stage="validation"
        )
        metrics, non_trainable_variables, metrics_variables = aux

        metrics_variables = self._update_metrics(loss, metrics_variables, self._batch_size_from_data(data))

        state = trainable_variables, non_trainable_variables, metrics_variables
        return metrics, state

    def stateless_train_step(self, state: tuple, data: dict[str, any]) -> (dict[str, jax.Array], tuple):
        """
        Stateless training step compatible with JAX autograd and stateless optimization.

        Computes gradients and applies optimizer updates in a purely functional style.

        Parameters
        ----------
        state : tuple
            Tuple of (trainable_variables, non_trainable_variables, optimizer_variables, metrics_variables).
        data : dict of str to any
            Input data for training.

        Returns
        -------
        metrics : dict of str to jax.Array
            Dictionary of computed training metrics.
        state : tuple
            Updated state tuple after training.
        """
        trainable_variables, non_trainable_variables, optimizer_variables, metrics_variables = state

        grad_fn = jax.value_and_grad(self.stateless_compute_metrics, has_aux=True)

        (loss, aux), grads = grad_fn(
            trainable_variables, non_trainable_variables, metrics_variables, data=data, stage="training"
        )
        metrics, non_trainable_variables, metrics_variables = aux

        trainable_variables, optimizer_variables = self.optimizer.stateless_apply(
            optimizer_variables, grads, trainable_variables
        )

        metrics_variables = self._update_metrics(loss, metrics_variables, self._batch_size_from_data(data))

        state = trainable_variables, non_trainable_variables, optimizer_variables, metrics_variables
        return metrics, state

    def test_step(self, *args, **kwargs):
        """
        Alias to `stateless_test_step` for compatibility with `keras.Model`.

        Parameters
        ----------
        *args, **kwargs : Any
            Passed through to `stateless_test_step`.

        Returns
        -------
        See `stateless_test_step`.
        """
        return self.stateless_test_step(*args, **kwargs)

    def train_step(self, *args, **kwargs):
        """
        Alias to `stateless_train_step` for compatibility with `keras.Model`.

        Parameters
        ----------
        *args, **kwargs : Any
            Passed through to `stateless_train_step`.

        Returns
        -------
        See `stateless_train_step`.
        """
        return self.stateless_train_step(*args, **kwargs)

    def _update_metrics(self, loss: jax.Array, metrics_variables: any, sample_weight: any = None) -> any:
        """
        Updates metric tracking variables in a stateless JAX-compatible way.

        This method updates the loss tracker (and any other Keras metrics)
        and returns updated metric variable states for downstream use.

        Parameters
        ----------
        loss : jax.Array
            Scalar loss used for metric tracking.
        metrics_variables : Any
            Current metric variable states.
        sample_weight : Any, optional
            Sample weights to apply during update.

        Returns
        -------
        metrics_variables : Any
            Updated metrics variable states.
        """
        state_mapping = list(zip(self.metrics_variables, metrics_variables))
        with keras.StatelessScope(state_mapping) as scope:
            self._loss_tracker.update_state(loss, sample_weight=sample_weight)

        # JAX is stateless, so we need to return the metrics as state in downstream functions
        metrics_variables = [scope.get_current_value(v) for v in self.metrics_variables]

        return metrics_variables

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
