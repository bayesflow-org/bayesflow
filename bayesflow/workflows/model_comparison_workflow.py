from collections.abc import Sequence

import time

import numpy as np

import keras

from bayesflow.networks import SummaryNetwork
from bayesflow.simulators import ModelComparisonSimulator, Simulator
from bayesflow.adapters import Adapter
from bayesflow.approximators import ModelComparisonApproximator
from bayesflow.utils import find_network, find_summary_network, logging, format_duration, filter_kwargs

from .basic_workflow import BasicWorkflow


class ModelComparisonWorkflow(BasicWorkflow):
    """
    This class extends :class:`~bayesflow.workflows.BasicWorkflow` to support
    amortized Bayesian model comparison.

    Parameters
    ----------
    simulator : Sequence[Simulator] or ModelComparisonSimulator, optional
        Either a list of per-model :class:`~bayesflow.simulators.Simulator` instances,
        which will be automatically wrapped in a
        :class:`~bayesflow.simulators.ModelComparisonSimulator` with uniform model
        priors, or a pre-built :class:`~bayesflow.simulators.ModelComparisonSimulator`.
    adapter : Adapter, optional
        Adapter for data processing. If not provided, a default adapter is built
        that maps ``model_indices`` to ``inference_variables`` and handles any
        ``inference_conditions`` / ``summary_variables``.
    classifier_network : keras.Layer or str, optional
        The classifier backbone used inside the approximator (default: ``"mlp"``).
        Accepts a Keras layer instance or any name recognised by
        :func:`~bayesflow.utils.find_network` (e.g. ``"mlp"``).
    summary_network : SummaryNetwork or str, optional
        Optional summary network for data compression (default: None).
    initial_learning_rate : float, optional
        Initial learning rate for the optimizer (default: 5e-4).
    optimizer : keras.optimizers.Optimizer or type, optional
        Optimizer instance or class. If None, a default schedule is built at
        fit time (default: None).
    checkpoint_filepath : str, optional
        Directory for saving model checkpoints (default: None).
    checkpoint_name : str, optional
        Base name for checkpoint files (default: ``"model"``).
    save_weights_only : bool, optional
        Save only weights rather than the full model (default: False).
    save_best_only : bool, optional
        Save only the best checkpoint instead of the last (default: False).
    inference_conditions : Sequence[str] or str, optional
        Keys in the simulator output to use as direct classifier conditions.
    summary_variables : Sequence[str] or str, optional
        Keys in the simulator output to compress via the summary network.
    model_names : Sequence[str], optional
        Human-readable names for each model, used in diagnostic plots.
    standardize : Sequence[str] or str or None, optional
        Variables to standardize. Defaults to ``None`` because model indices
        are one-hot encoded and should not be standardized.
    **kwargs : dict, optional
        Additional keyword arguments organised by context:

        - ``classifier_kwargs`` : dict — passed to :func:`~bayesflow.utils.find_network`.
        - ``summary_kwargs``    : dict — passed to :func:`~bayesflow.utils.find_summary_network`.
        - ``optimizer_kwargs``  : dict — passed to ``_init_optimizer``.
        - ``simulator_kwargs``  : dict — passed to :class:`~bayesflow.simulators.ModelComparisonSimulator`.
        - Other keys forwarded to :class:`~bayesflow.approximators.ModelComparisonApproximator`.
    """

    def __init__(
        self,
        simulator: Sequence[Simulator] | ModelComparisonSimulator | None = None,
        adapter: Adapter | None = None,
        classifier_network: keras.Layer | str = "mlp",
        summary_network: SummaryNetwork | str | None = None,
        initial_learning_rate: float = 5e-4,
        optimizer: keras.optimizers.Optimizer | type | None = None,
        checkpoint_filepath: str | None = None,
        checkpoint_name: str = "model",
        save_weights_only: bool = False,
        save_best_only: bool = False,
        inference_conditions: Sequence[str] | str | None = None,
        summary_variables: Sequence[str] | str | None = None,
        model_names: Sequence[str] | None = None,
        standardize: Sequence[str] | str | None = None,
        **kwargs,
    ):
        if isinstance(simulator, Sequence):
            simulator = ModelComparisonSimulator(
                simulators=simulator,
                **kwargs.pop("simulator_kwargs", {}),
            )

        if simulator is not None and len(simulator.simulators) < 2:
            raise ValueError(
                f"ModelComparisonWorkflow requires at least 2 simulators, got {len(simulator.simulators)}."
            )

        self.simulator = simulator
        self.model_names = model_names

        num_models = len(simulator.simulators) if simulator is not None else None

        adapter = adapter or ModelComparisonWorkflow.default_adapter(inference_conditions, summary_variables)

        self.approximator = ModelComparisonApproximator(
            num_models=num_models,
            classifier_network=find_network(classifier_network, **kwargs.get("classifier_kwargs", {})),
            summary_network=find_summary_network(summary_network, **kwargs.get("summary_kwargs", {})),
            adapter=adapter,
            standardize=standardize,
            **filter_kwargs(kwargs, ModelComparisonApproximator.__init__),
        )

        self._init_optimizer(initial_learning_rate, optimizer, **kwargs.get("optimizer_kwargs", {}))
        self._init_checkpointing(checkpoint_filepath, checkpoint_name, save_weights_only, save_best_only)
        self.history = None
        self._needs_compile = True

    @property
    def classifier_network(self) -> keras.Layer:
        """The classifier backbone (subnet of the internal ScoringRuleNetwork)."""
        return self.approximator.inference_network.subnet

    @staticmethod
    def default_adapter(
        inference_conditions: Sequence[str] | str | None,
        summary_variables: Sequence[str] | str | None,
    ) -> Adapter:
        """
        Build a default adapter for model comparison data.

        Maps the ``model_indices`` key produced by
        :class:`~bayesflow.simulators.ModelComparisonSimulator` to
        ``inference_variables``, and optionally concatenates condition and
        summary keys.

        Parameters
        ----------
        inference_conditions : Sequence[str] or str or None
            Keys to concatenate into ``inference_conditions``.
        summary_variables : Sequence[str] or str or None
            Keys to concatenate into ``summary_variables``.

        Returns
        -------
        Adapter
        """
        adapter = (
            Adapter()
            .convert_dtype(from_dtype="float64", to_dtype="float32")
            .concatenate("model_indices", into="inference_variables")
        )

        if inference_conditions is not None:
            adapter = adapter.concatenate(inference_conditions, into="inference_conditions")
        if summary_variables is not None:
            adapter = adapter.concatenate(summary_variables, into="summary_variables")

        return adapter

    def predict(
        self,
        *,
        conditions: dict,
        probs: bool = True,
        **kwargs,
    ) -> np.ndarray:
        """
        Return posterior model probabilities (or logits) for the given conditions.

        Parameters
        ----------
        conditions : dict[str, np.ndarray]
            Conditioning data as produced by the simulator (or real observations).
        probs : bool, optional
            Return softmax probabilities when True (default), raw logits when False.
        **kwargs
            Forwarded to :meth:`~bayesflow.approximators.ModelComparisonApproximator.predict`.

        Returns
        -------
        np.ndarray of shape (num_datasets, num_models)
            Posterior model probabilities (or logits).
        """
        start_time = time.perf_counter()
        predictions = self.approximator.predict(conditions=conditions, probs=probs, **kwargs)
        elapsed = time.perf_counter() - start_time
        logging.info(f"Prediction completed in {format_duration(elapsed)}.")
        return predictions

    # ------------------------------------------------------------------
    # Disable BasicWorkflow inference methods that do not apply here
    # ------------------------------------------------------------------

    def sample(self, *args, **kwargs):
        raise NotImplementedError("ModelComparisonWorkflow does not support sampling. Use predict() instead.")

    def estimate(self, *args, **kwargs):
        raise NotImplementedError("ModelComparisonWorkflow does not support estimate(). Use predict() instead.")

    def log_prob(self, *args, **kwargs):
        raise NotImplementedError("ModelComparisonWorkflow does not support log_prob().")

    def ancestral_sample(self, *args, **kwargs):
        raise NotImplementedError("ModelComparisonWorkflow does not support ancestral_sample(). Use predict() instead.")
