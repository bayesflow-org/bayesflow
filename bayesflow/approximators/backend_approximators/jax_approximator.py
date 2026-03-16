from typing import Any

import jax
import keras

from bayesflow.utils import filter_kwargs


class JAXApproximator(keras.Model):
    """Backend-specific base for the JAX stateless training loop.

    JAX requires a purely functional style: all state (trainable
    weights, non-trainable variables, optimizer variables, metric
    variables) is threaded explicitly through each step via
    :class:`keras.StatelessScope`.

    Key methods:

    - :meth:`stateless_compute_metrics` — the ``value_and_grad`` target.
    - :meth:`stateless_train_step` / :meth:`stateless_test_step` —
      orchestrate one optimisation / validation step.
    - :meth:`train_step` / :meth:`test_step` — aliases required by
      :meth:`keras.Model.fit`.

    Subclasses must implement
    :meth:`compute_metrics` and :meth:`_batch_size_from_data`
    (see :class:`BackendApproximator` for details).
    """

    # noinspection PyMethodOverriding
    def compute_metrics(self, *args, **kwargs) -> dict[str, jax.Array]:
        raise NotImplementedError

    def stateless_compute_metrics(
        self,
        trainable_variables: Any,
        non_trainable_variables: Any,
        metrics_variables: Any,
        data: dict[str, Any],
        stage: str = "training",
    ) -> tuple[jax.Array, tuple]:
        """Stateless forward pass used as the ``jax.value_and_grad`` target.

        All model state is injected via :class:`keras.StatelessScope` so
        that JAX can differentiate through the computation.

        Parameters
        ----------
        trainable_variables : Any
            Current trainable weight values.
        non_trainable_variables : Any
            Current non-trainable variable values (e.g. batch-norm statistics).
        metrics_variables : Any
            Current metric tracking variable values.
        data : dict[str, Any]
            Input data dictionary passed to :meth:`compute_metrics`.
        stage : str, default ``"training"``
            ``"training"`` or ``"validation"``.

        Returns
        -------
        loss : jax.Array
            Scalar loss for gradient computation.
        aux : tuple
            ``(metrics_dict, updated_non_trainable_variables,
            updated_metrics_variables)``.
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

    def stateless_test_step(self, state: tuple, data: dict[str, Any]) -> tuple[dict[str, jax.Array], tuple]:
        """Stateless validation step.

        Parameters
        ----------
        state : tuple
            ``(trainable_variables, non_trainable_variables, metrics_variables)``.
        data : dict[str, Any]
            Input data for validation.

        Returns
        -------
        metrics : dict[str, jax.Array]
            Computed evaluation metrics.
        state : tuple
            Updated state tuple.
        """
        trainable_variables, non_trainable_variables, metrics_variables = state

        loss, aux = self.stateless_compute_metrics(
            trainable_variables, non_trainable_variables, metrics_variables, data=data, stage="validation"
        )
        metrics, non_trainable_variables, metrics_variables = aux

        metrics_variables = self._update_metrics(loss, metrics_variables, self._batch_size_from_data(data))

        state = trainable_variables, non_trainable_variables, metrics_variables
        return metrics, state

    def stateless_train_step(self, state: tuple, data: dict[str, Any]) -> tuple[dict[str, jax.Array], tuple]:
        """Stateless training step with ``jax.value_and_grad``.

        Computes gradients via ``jax.value_and_grad`` on
        :meth:`stateless_compute_metrics` and applies the optimizer
        update statelessly.

        Parameters
        ----------
        state : tuple
            ``(trainable_variables, non_trainable_variables,
            optimizer_variables, metrics_variables)``.
        data : dict[str, Any]
            Input data for training.

        Returns
        -------
        metrics : dict[str, jax.Array]
            Computed training metrics.
        state : tuple
            Updated state tuple.
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
        """Alias for :meth:`stateless_test_step` (required by :meth:`keras.Model.fit`)."""
        return self.stateless_test_step(*args, **kwargs)

    def train_step(self, *args, **kwargs):
        """Alias for :meth:`stateless_train_step` (required by :meth:`keras.Model.fit`)."""
        return self.stateless_train_step(*args, **kwargs)

    def _update_metrics(self, loss: jax.Array, metrics_variables: Any, sample_weight: Any = None) -> Any:
        """Stateless metric update for JAX.

        Enters a :class:`keras.StatelessScope` to update the loss
        tracker, then extracts and returns the new metric variable
        states.

        Parameters
        ----------
        loss : jax.Array
            Scalar loss value.
        metrics_variables : Any
            Current metric variable states.
        sample_weight : Any, optional
            Optional sample weights.

        Returns
        -------
        Any
            Updated metrics variable states.
        """
        state_mapping = list(zip(self.metrics_variables, metrics_variables))
        with keras.StatelessScope(state_mapping) as scope:
            self._loss_tracker.update_state(loss, sample_weight=sample_weight)

        # JAX is stateless, so we need to return the metrics as state in downstream functions
        metrics_variables = [scope.get_current_value(v) for v in self.metrics_variables]

        return metrics_variables

    # noinspection PyMethodOverriding
    def _batch_size_from_data(self, data: Any) -> int:
        raise NotImplementedError(
            "Correct calculation of the metrics requires obtaining the batch size from the supplied data "
            "for proper weighting of metrics for batches with different sizes. Please implement the "
            "_batch_size_from_data method for your approximator. For a given batch of data, it should "
            "return the corresponding batch size."
        )
