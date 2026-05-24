from collections.abc import Callable, Mapping, Sequence

import time

import numpy as np

import keras

import matplotlib.pyplot as plt

from bayesflow.networks import SummaryNetwork
from bayesflow.simulators import ModelComparisonSimulator, Simulator
from bayesflow.adapters import Adapter
from bayesflow.approximators import ModelComparisonApproximator
from bayesflow.scoring_rules import CrossEntropyScore, ScoringRule
from bayesflow.utils import find_network, find_summary_network, logging, format_duration, filter_kwargs
from bayesflow.diagnostics import plots as bf_plots
from bayesflow.diagnostics import metrics as bf_metrics

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
    scoring_rule : ScoringRule, optional
        Scoring rule used to train the classifier. Determines what the network
        learns to estimate:

        - **PMP rules** (:class:`~bayesflow.scoring_rules.CrossEntropyScore` (default),
          :class:`~bayesflow.scoring_rules.SquaredScore`,
          :class:`~bayesflow.scoring_rules.PolynomialScore`): network outputs softmax
          probabilities over all ``num_models`` models.
        - **Bayes factor rules** (:class:`~bayesflow.scoring_rules.ExponentialScore`,
          :class:`~bayesflow.scoring_rules.LogisticScore`,
          :class:`~bayesflow.scoring_rules.LPOPExponentialScore`, etc.): network outputs
          ``num_models - 1`` log Bayes factors relative to model 0.
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
        When ``shared_simulator`` is provided, these keys are automatically
        broadcast to the batch dimension before being passed to the network.
    summary_variables : Sequence[str] or str, optional
        Keys in the simulator output to compress via the summary network.
    shared_simulator : Simulator or Callable, optional
        A shared simulator that provides context variables to every per-model
        simulator (e.g. sample size ``n``). When ``simulator`` is a list this
        is forwarded to :class:`~bayesflow.simulators.ModelComparisonSimulator`.
        Scalar outputs from the shared simulator that are listed in
        ``inference_conditions`` are broadcast to the batch dimension automatically.
    use_mixed_batches : bool, optional
        Whether each training batch mixes samples from different models
        (default: ``True``). Forwarded to
        :class:`~bayesflow.simulators.ModelComparisonSimulator` when ``simulator``
        is a list.
    model_names : Sequence[str], optional
        Human-readable names for each model, used in diagnostic plots.
    standardize : Sequence[str] or str or None, optional
        Variables to standardize. When a ``summary_network`` is provided and
        this argument is ``None``, defaults to ``["summary_variables"]`` so
        that the raw data is standardized before entering the summary network.
        ``inference_variables`` (one-hot model indices) are never standardized.
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
        scoring_rule: ScoringRule | None = None,
        initial_learning_rate: float = 5e-4,
        optimizer: keras.optimizers.Optimizer | type | None = None,
        checkpoint_filepath: str | None = None,
        checkpoint_name: str = "model",
        save_weights_only: bool = False,
        save_best_only: bool = False,
        inference_conditions: Sequence[str] | str | None = None,
        summary_variables: Sequence[str] | str | None = None,
        shared_simulator: Simulator | Callable | None = None,
        use_mixed_batches: bool = True,
        model_names: Sequence[str] | None = None,
        standardize: Sequence[str] | str | None = None,
        **kwargs,
    ):
        if isinstance(simulator, Sequence):
            simulator = ModelComparisonSimulator(
                simulators=simulator,
                shared_simulator=shared_simulator,
                use_mixed_batches=use_mixed_batches,
                **kwargs.pop("simulator_kwargs", {}),
            )

        if simulator is not None and len(simulator.simulators) < 2:
            raise ValueError(
                f"ModelComparisonWorkflow requires at least 2 simulators, got {len(simulator.simulators)}."
            )

        self.simulator = simulator
        self.model_names = model_names

        num_models = len(simulator.simulators) if simulator is not None else None

        # When a shared_simulator provides context variables used as inference_conditions,
        # those variables are scalars (one value per batch, not per sample) and must be
        # broadcast to the batch dimension before the adapter can concatenate them.
        # We auto-detect the broadcast target as the first summary variable.
        broadcast_ref = None
        if shared_simulator is not None and inference_conditions is not None:
            sv = [summary_variables] if isinstance(summary_variables, str) else (summary_variables or [])
            if sv:
                broadcast_ref = sv[0]

        adapter = adapter or ModelComparisonWorkflow.default_adapter(
            inference_conditions=inference_conditions,
            summary_variables=summary_variables,
            broadcast_conditions_to=broadcast_ref,
        )

        # When a summary network is present and the caller did not specify standardize,
        # default to standardizing summary_variables.  inference_variables (model indices)
        # are one-hot and must never be standardized.
        if standardize is None and summary_network is not None:
            standardize = ["summary_variables"]

        self.approximator = ModelComparisonApproximator(
            num_models=num_models,
            classifier_network=find_network(classifier_network, **kwargs.get("classifier_kwargs", {})),
            summary_network=find_summary_network(summary_network, **kwargs.get("summary_kwargs", {})),
            scoring_rule=scoring_rule if scoring_rule is not None else CrossEntropyScore(),
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
        broadcast_conditions_to: str | None = None,
    ) -> Adapter:
        """
        Build a default adapter for model comparison data.

        Maps the ``model_indices`` key produced by
        :class:`~bayesflow.simulators.ModelComparisonSimulator` to
        ``inference_variables``, and optionally handles condition and summary keys.

        Parameters
        ----------
        inference_conditions : Sequence[str] or str or None
            Keys to concatenate into ``inference_conditions``.
        summary_variables : Sequence[str] or str or None
            Keys to concatenate into ``summary_variables``.
        broadcast_conditions_to : str or None
            If provided, each ``inference_conditions`` key is broadcast to the
            batch dimension of this variable before any renaming. Used when a
            ``shared_simulator`` returns scalar context variables.

        Returns
        -------
        Adapter
        """
        adapter = (
            Adapter()
            .convert_dtype(from_dtype="float64", to_dtype="float32")
            .concatenate("model_indices", into="inference_variables")
        )

        # Broadcast scalar context variables (from shared_simulator) to batch dim.
        # Must happen before as_set so the broadcast target still has its original name.
        if broadcast_conditions_to is not None and inference_conditions is not None:
            conds = [inference_conditions] if isinstance(inference_conditions, str) else list(inference_conditions)
            adapter = adapter.broadcast(conds, to=broadcast_conditions_to)

        if summary_variables is not None:
            adapter = adapter.concatenate(summary_variables, into="summary_variables")

        if inference_conditions is not None:
            adapter = adapter.concatenate(inference_conditions, into="inference_conditions")

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
            When ``True`` (default) always returns posterior model probabilities of shape
            ``(num_datasets, num_models)``.  For PMP rules these come from softmax over
            logits; for Bayes factor rules the reference anchor :math:`f_0 = 0` is
            prepended before softmax (assumes equal model priors).
            When ``False`` returns raw logits (PMP rules, shape ``(N, M)``) or raw log
            Bayes factors relative to model 0 (Bayes factor rules, shape ``(N, M-1)``).
        **kwargs
            Forwarded to :meth:`~bayesflow.approximators.ModelComparisonApproximator.predict`.

        Returns
        -------
        np.ndarray
            Shape ``(num_datasets, num_models)`` when ``probs=True`` (always).
            Shape ``(num_datasets, num_models)`` when ``probs=False`` with a PMP rule.
            Shape ``(num_datasets, num_models - 1)`` when ``probs=False`` with a Bayes
            factor rule.
        """
        start_time = time.perf_counter()
        predictions = self.approximator.predict(conditions=conditions, probs=probs, **kwargs)
        elapsed = time.perf_counter() - start_time
        logging.info(f"Prediction completed in {format_duration(elapsed)}.")
        return predictions

    def plot_default_diagnostics(
        self,
        test_data: Mapping[str, np.ndarray] | int,
        true_log_bfs_fn: callable = None,
        **kwargs,
    ) -> dict[str, plt.Figure]:
        r"""
        Generate default diagnostic plots for model comparison.

        Produces a loss curve (when training history is available) followed by
        a set of plots that depend on the active scoring rule:

        **PMP scoring rules** (:class:`~bayesflow.scoring_rules.CrossEntropyScore`,
        :class:`~bayesflow.scoring_rules.SquaredScore`,
        :class:`~bayesflow.scoring_rules.PolynomialScore`):

        - ``"confusion_matrix"`` — posterior model probability confusion matrix.
        - ``"calibration"`` — per-model calibration curves with ECE annotations.

        **Bayes factor scoring rules** (:class:`~bayesflow.scoring_rules.ExponentialScore`,
        :class:`~bayesflow.scoring_rules.LogisticScore`,
        :class:`~bayesflow.scoring_rules.LPOPExponentialScore`, etc.):

        - ``"blind_coverage"`` — blind coverage test (Jeffrey & Wandelt 2024):
          conditional ECDFs of predicted log Bayes factors stratified by true model,
          evaluated against blind (model-label-free) quantile thresholds.
        - ``"pairwise_bayes_factors"`` — heatmap of the mean predicted
          :math:`\log K_{m,j}` stratified by true model, showing pairwise
          model separability across all :math:`M \times M` pairs.
        - ``"bayes_factor_recovery"`` — scatter of predicted vs. true log Bayes
          factors, one panel per competing model.  Only produced when
          ``true_log_bfs_fn`` is supplied.

        Parameters
        ----------
        test_data : Mapping[str, np.ndarray] or int
            Either a pre-simulated data dictionary (as returned by
            :meth:`simulate`) or an integer specifying how many datasets to
            generate using the attached simulator.
        true_log_bfs_fn : callable or None, optional
            A function ``(test_data: dict) -> np.ndarray`` that receives the
            simulated data dictionary and returns ground-truth log Bayes factors
            of shape ``(num_datasets, num_models - 1)``.  When provided, a
            ``"bayes_factor_recovery"`` plot is added for Bayes factor scoring
            rules.  Ignored for PMP scoring rules.
        **kwargs : dict, optional
            Fine-grained control over individual plots via nested dicts:

            - ``test_data_kwargs`` — forwarded to :meth:`simulate` when
              ``test_data`` is an integer.
            - ``predict_kwargs`` — forwarded to :meth:`predict`.
            - ``loss_kwargs`` — forwarded to :func:`~bayesflow.diagnostics.plots.loss`.
            - ``confusion_matrix_kwargs`` — forwarded to
              :func:`~bayesflow.diagnostics.plots.mc_confusion_matrix` (PMP only).
            - ``calibration_kwargs`` — forwarded to
              :func:`~bayesflow.diagnostics.plots.mc_calibration` (PMP only).
            - ``blind_coverage_kwargs`` — forwarded to
              :func:`~bayesflow.diagnostics.plots.blind_coverage` (BF scoring rules only).
            - ``bayes_factor_recovery_kwargs`` — forwarded to
              :func:`~bayesflow.diagnostics.plots.bayes_factor_recovery` (BF scoring
              rules only, requires ``true_log_bfs_fn``).
            - ``pairwise_bayes_factors_kwargs`` — forwarded to
              :func:`~bayesflow.diagnostics.plots.pairwise_bayes_factors` (BF scoring
              rules only).

        Returns
        -------
        dict[str, plt.Figure]
            Keys are plot names; values are the corresponding
            :class:`~matplotlib.figure.Figure` objects.

        Raises
        ------
        ValueError
            If ``test_data`` is an integer but no simulator is attached.
        """
        if isinstance(test_data, int):
            if self.simulator is None:
                raise ValueError(f"No simulator attached. Cannot generate {test_data} test datasets.")
            test_data = self.simulate(test_data, **kwargs.get("test_data_kwargs", {}))

        figures = {}

        if self.history is not None:
            figures["loss"] = bf_plots.loss(self.history, **kwargs.get("loss_kwargs", {}))

        if "model_indices" in test_data:
            true_models = test_data["model_indices"]
        elif "inference_variables" in test_data:
            true_models = test_data["inference_variables"]
        else:
            raise KeyError(
                "test_data must contain 'model_indices' (raw simulator output) or "
                "'inference_variables' (adapted output). Neither key was found."
            )

        # Determine mode before calling predict so we can request the right output format:
        # PMP diagnostic plots need probs=True (softmax probabilities);
        # BF diagnostic plots need probs=False (raw log Bayes factors).
        head_shapes = self.approximator.scoring_rule.get_head_shapes_from_target_shape((1, 2))
        is_pmp_mode = "logits" in head_shapes

        predict_kwargs = dict(kwargs.get("predict_kwargs", {}))
        predict_kwargs.setdefault("probs", is_pmp_mode)
        predictions = self.predict(conditions=test_data, **predict_kwargs)

        if is_pmp_mode:
            figures["confusion_matrix"] = bf_plots.mc_confusion_matrix(
                pred_models=predictions,
                true_models=true_models,
                model_names=self.model_names,
                **kwargs.get("confusion_matrix_kwargs", {}),
            )
            figures["calibration"] = bf_plots.mc_calibration(
                pred_models=predictions,
                true_models=true_models,
                model_names=self.model_names,
                **kwargs.get("calibration_kwargs", {}),
            )
        else:
            figures["blind_coverage"] = bf_plots.blind_coverage(
                pred_log_bayes_factors=predictions,
                true_models=true_models,
                model_names=self.model_names,
                **kwargs.get("blind_coverage_kwargs", {}),
            )
            figures["pairwise_bayes_factors"] = bf_plots.pairwise_bayes_factors(
                pred_log_bayes_factors=predictions,
                true_models=true_models,
                model_names=self.model_names,
                **kwargs.get("pairwise_bayes_factors_kwargs", {}),
            )
            if true_log_bfs_fn is not None:
                true_log_bfs = true_log_bfs_fn(test_data)
                figures["bayes_factor_recovery"] = bf_plots.bayes_factor_recovery(
                    pred_log_bayes_factors=predictions,
                    true_log_bayes_factors=true_log_bfs,
                    true_models=true_models,
                    model_names=self.model_names,
                    **kwargs.get("bayes_factor_recovery_kwargs", {}),
                )

        return figures

    def compute_default_diagnostics(
        self,
        test_data: Mapping[str, np.ndarray] | int,
        **kwargs,
    ) -> dict[str, any]:
        """
        Compute default scalar diagnostic metrics for model comparison.

        For **PMP scoring rules**:

        - ``"accuracy"`` — fraction of datasets for which ``argmax(PMPs)`` matches
          the true model.
        - ``"ece"`` — per-model expected calibration errors, as returned by
          :func:`~bayesflow.diagnostics.metrics.expected_calibration_error`.

        For **Bayes factor scoring rules**:

        - ``"accuracy"`` — fraction of datasets for which
          ``argmax(0, f_1, …, f_{M-1})`` matches the true model.

        Parameters
        ----------
        test_data : Mapping[str, np.ndarray] or int
            Either a pre-simulated data dictionary or an integer specifying how many
            datasets to generate using the attached simulator.
        **kwargs : dict, optional
            Fine-grained control via nested dicts:

            - ``test_data_kwargs`` — forwarded to :meth:`simulate` when ``test_data``
              is an integer.
            - ``predict_kwargs`` — forwarded to :meth:`predict`.
            - ``ece_kwargs`` — forwarded to
              :func:`~bayesflow.diagnostics.metrics.expected_calibration_error`
              (PMP mode only).

        Returns
        -------
        dict[str, any]
            Dictionary of diagnostic metrics.

        Raises
        ------
        ValueError
            If ``test_data`` is an integer but no simulator is attached.
        """
        if isinstance(test_data, int):
            if self.simulator is None:
                raise ValueError(f"No simulator attached. Cannot generate {test_data} test datasets.")
            test_data = self.simulate(test_data, **kwargs.get("test_data_kwargs", {}))

        predictions = self.predict(conditions=test_data, **kwargs.get("predict_kwargs", {}))

        if "model_indices" in test_data:
            true_models = test_data["model_indices"]
        elif "inference_variables" in test_data:
            true_models = test_data["inference_variables"]
        else:
            raise KeyError(
                "test_data must contain 'model_indices' (raw simulator output) or "
                "'inference_variables' (adapted output). Neither key was found."
            )

        head_shapes = self.approximator.scoring_rule.get_head_shapes_from_target_shape((1, true_models.shape[-1]))
        is_pmp_mode = "logits" in head_shapes

        # predict(probs=True) always returns shape (N, M) PMPs — suitable for accuracy in both modes
        metrics = {"accuracy": bf_metrics.model_comparison_accuracy(predictions, true_models)}

        if is_pmp_mode:
            ece_result = bf_metrics.expected_calibration_error(
                estimates=predictions,
                targets=true_models,
                model_names=self.model_names,
                **kwargs.get("ece_kwargs", {}),
            )
            metrics["ece"] = ece_result

        return metrics

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
