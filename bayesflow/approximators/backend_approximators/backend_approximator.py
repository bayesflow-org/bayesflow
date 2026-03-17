import keras

from bayesflow.utils import filter_kwargs


match keras.backend.backend():
    case "jax":
        from .jax_approximator import JAXApproximator as BaseBackendApproximator
    case "tensorflow":
        from .tensorflow_approximator import TensorFlowApproximator as BaseBackendApproximator
    case "torch":
        from .torch_approximator import TorchApproximator as BaseBackendApproximator
    case other:
        raise ValueError(f"Backend '{other}' is not supported.")


class BackendApproximator(BaseBackendApproximator):
    """Backend-agnostic base class for all BayesFlow approximators.

    Dynamically inherits from the backend-specific approximator
    (:class:`JAXApproximator`, :class:`TensorFlowApproximator`, or
    :class:`TorchApproximator`) depending on the active Keras backend,
    providing a unified training interface.

    Subclasses must implement:

    - :meth:`compute_metrics` — return a ``dict[str, Tensor]`` containing
      at least a ``"loss"`` key.
    - :meth:`_batch_size_from_data` — return the batch size for a given
      data dict so that metrics are correctly weighted across batches
      of different sizes.
    """

    # noinspection PyMethodOverriding
    def fit(self, *, dataset: keras.utils.PyDataset, **kwargs):
        """Train the model on the given dataset.

        Thin wrapper around :meth:`keras.Model.fit` that maps the
        BayesFlow ``dataset`` keyword to the Keras ``x`` positional
        argument and filters unsupported keyword arguments.

        Parameters
        ----------
        dataset : keras.utils.PyDataset
            A Keras-compatible dataset that yields batches of data dicts.
        **kwargs
            Additional keyword arguments forwarded to :meth:`keras.Model.fit`
            (e.g. ``epochs``, ``callbacks``, ``validation_data``).

        Returns
        -------
        keras.callbacks.History
            A Keras ``History`` object containing training metrics.
        """
        return super().fit(x=dataset, y=None, **filter_kwargs(kwargs, super().fit))
