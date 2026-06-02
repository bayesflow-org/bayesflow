from collections.abc import Callable, Mapping, Sequence
import time

import numpy as np
import matplotlib.pyplot as plt

import keras

from bayesflow.networks import SummaryNetwork
from bayesflow.simulators import ModelComparisonSimulator, Simulator
from bayesflow.adapters import Adapter
from bayesflow.approximators import ModelComparisonApproximator
from bayesflow.scoring_rules import CategoricalScoringRule
from bayesflow.utils import (
    find_network,
    find_scoring_rule,
    find_summary_network,
    logging,
    format_duration,
    filter_kwargs,
)
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
    scoring_rules : CategoricalScoringRule or dict[str, CategoricalScoringRule] or str, optional
        Scoring rule(s) used to train the classifier. Accepts a single
        :class:`~bayesflow.scoring_rules.CategoricalScoringRule` instance, a string recognised by
        :func:`~bayesflow.utils.find_scoring_rule`, or a mapping of rule names to rules for
        co-learning multiple scoring rules simultaneously (default: ``"cross_entropy"``).
        Determines what the network learns to estimate:

        - **PMP rules** (``"cross_entropy"``, ``"brier"``, ``"polynomial"``): network
          outputs softmax probabilities over all ``num_models`` models.
        - **Bayes factor rules** (``"exponential"``, ``"scaled_exponential"``,
          ``"leaky_exponential"``, ``"logistic"``, ``"power_logistic"``): network outputs
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
        simulator: Sequence[Simulator] | ModelComparisonSimulator = None,
        adapter: Adapter = None,
        classifier_network: keras.Layer | str = "mlp",
        summary_network: SummaryNetwork | str = None,
        scoring_rules: CategoricalScoringRule | dict[str, CategoricalScoringRule] | str = None,
        initial_learning_rate: float = 5e-4,
        optimizer: keras.optimizers.Optimizer | type = None,
        checkpoint_filepath: str = None,
        checkpoint_name: str = "model",
        save_weights_only: bool = False,
        save_best_only: bool = False,
        inference_conditions: Sequence[str] | str = None,
        summary_variables: Sequence[str] | str = None,
        shared_simulator: Simulator | Callable = None,
        use_mixed_batches: bool = True,
        model_names: Sequence[str] = None,
        standardize: Sequence[str] | str = None,
        **kwargs,
    ):
        if isinstance(simulator, Sequence):
            simulator = ModelComparisonSimulator(
                simulators=simulator,
                shared_simulator=shared_simulator,
                use_mixed_batches=use_mixed_batches,
                **kwargs.get("simulator_kwargs", {}),
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

        if standardize is None and summary_network is not None:
            standardize = ["summary_variables"]

        self.approximator = ModelComparisonApproximator(
            num_models=num_models,
            classifier_network=find_network(classifier_network, **kwargs.get("classifier_kwargs", {})),
            summary_network=find_summary_network(summary_network, **kwargs.get("summary_kwargs", {})),
            scoring_rules=find_scoring_rule(scoring_rules) if isinstance(scoring_rules, str) else scoring_rules,
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
        inference_conditions: Sequence[str] | str,
        summary_variables: Sequence[str] | str,
        broadcast_conditions_to: str = None,
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
        # Must happen before concatenate so the broadcast target still has its original name.
        if broadcast_conditions_to is not None and inference_conditions is not None:
            conds = [inference_conditions] if isinstance(inference_conditions, str) else list(inference_conditions)
            adapter = adapter.broadcast(conds, to=broadcast_conditions_to)

        if summary_variables is not None:
            adapter = adapter.concatenate(summary_variables, into="summary_variables")

        if inference_conditions is not None:
            adapter = adapter.concatenate(inference_conditions, into="inference_conditions")

        return adapter

    def estimate(
        self,
        *,
        conditions: dict,
        **kwargs,
    ) -> dict[str, np.ndarray]:
        """
        Return posterior model probabilities (and raw network outputs) for the given conditions.

        Parameters
        ----------
        conditions : dict[str, np.ndarray]
            Conditioning data as produced by the simulator (or real observations).
        **kwargs
            Forwarded to :meth:`~bayesflow.approximators.ModelComparisonApproximator.estimate`.

        Returns
        -------
        dict[str, np.ndarray]
            Always contains ``"model_probs"`` of shape ``(num_datasets, num_models)``.
            PMP rules additionally contain ``"logits"`` of shape ``(num_datasets, num_models)``.
            Bayes factor rules additionally contain ``"log_bayes_factors"`` of shape
            ``(num_datasets, num_models - 1)``.
        """
        start_time = time.perf_counter()
        estimates = self.approximator.estimate(conditions=conditions, **kwargs)
        elapsed = time.perf_counter() - start_time
        logging.info(f"Estimation completed in {format_duration(elapsed)}.")
        return estimates

    def plot_default_diagnostics(
        self,
        test_data: Mapping[str, np.ndarray] | int,
        true_log_bfs_fn: Callable | None = None,
        **kwargs,
    ) -> dict[str, plt.Figure]:
        r"""
        Generate default diagnostic plots for model comparison.

        Produces a loss curve (when training history is available) followed by
        a set of plots that depend on the active scoring rule:

        **PMP scoring rules** (:class:`~bayesflow.scoring_rules.CrossEntropyScore`,
        :class:`~bayesflow.scoring_rules.BrierScore`,
        :class:`~bayesflow.scoring_rules.PolynomialScore`):

        - ``"confusion_matrix"`` — posterior model probability confusion matrix.
        - ``"calibration"`` — per-model calibration curves with ECE annotations.

        **Bayes factor scoring rules** (:class:`~bayesflow.scoring_rules.ExponentialScore`,
        :class:`~bayesflow.scoring_rules.LogisticScore`):

        - ``"calibration"`` — per-model calibration curves (Jeffrey & Wandelt 2024
          blind coverage): predicted posterior model probability on x, observed
          fraction on y, with ECE annotations.
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
              :func:`~bayesflow.diagnostics.plots.mc_calibration` (both PMP and BF rules).
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

        scoring_rules = self.approximator.inference_network.scoring_rules
        if len(scoring_rules) > 1:
            raise NotImplementedError(
                "Default diagnostics for multiple scoring rules are not yet implemented. "
                "Use approximator.estimate() to obtain a dict of per-rule estimates keyed by rule name, "
                "then pass each rule's 'model_probs' (and 'logits' or 'log_bayes_factors') "
                "to the bayesflow.diagnostics functions directly."
            )
        is_pmp_mode = next(iter(scoring_rules.values())).is_pmp_rule

        estimates = self.estimate(conditions=test_data, **kwargs.get("estimate_kwargs", {}))

        if is_pmp_mode:
            figures["confusion_matrix"] = bf_plots.mc_confusion_matrix(
                pred_models=estimates["model_probs"],
                true_models=true_models,
                model_names=self.model_names,
                **kwargs.get("confusion_matrix_kwargs", {}),
            )
            figures["calibration"] = bf_plots.mc_calibration(
                pred_models=estimates["model_probs"],
                true_models=true_models,
                model_names=self.model_names,
                **kwargs.get("calibration_kwargs", {}),
            )
        else:
            bf_calibration_defaults = dict(color="#21908c")
            figures["calibration"] = bf_plots.mc_calibration(
                pred_models=estimates["model_probs"],
                true_models=true_models,
                model_names=self.model_names,
                **{**bf_calibration_defaults, **kwargs.get("calibration_kwargs", {})},
            )
            figures["pairwise_bayes_factors"] = bf_plots.pairwise_bayes_factors(
                pred_log_bayes_factors=estimates["log_bayes_factors"],
                true_models=true_models,
                model_names=self.model_names,
                **kwargs.get("pairwise_bayes_factors_kwargs", {}),
            )
            if true_log_bfs_fn is not None:
                true_log_bfs = true_log_bfs_fn(test_data)
                figures["bayes_factor_recovery"] = bf_plots.bayes_factor_recovery(
                    pred_log_bayes_factors=estimates["log_bayes_factors"],
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
    ) -> dict:
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
            - ``estimate_kwargs`` — forwarded to :meth:`estimate`.
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

        estimates = self.estimate(conditions=test_data, **kwargs.get("estimate_kwargs", {}))

        if "model_indices" in test_data:
            true_models = test_data["model_indices"]
        elif "inference_variables" in test_data:
            true_models = test_data["inference_variables"]
        else:
            raise KeyError(
                "test_data must contain 'model_indices' (raw simulator output) or "
                "'inference_variables' (adapted output). Neither key was found."
            )

        scoring_rules = self.approximator.inference_network.scoring_rules
        if len(scoring_rules) > 1:
            raise NotImplementedError(
                "Default diagnostics for multiple scoring rules are not yet implemented. "
                "Use approximator.estimate() to obtain a dict of per-rule estimates keyed by rule name, "
                "then pass each rule's 'model_probs' (and 'logits' or 'log_bayes_factors') "
                "to the bayesflow.diagnostics functions directly."
            )
        is_pmp_mode = next(iter(scoring_rules.values())).is_pmp_rule

        metrics = {"accuracy": bf_metrics.model_comparison_accuracy(estimates["model_probs"], true_models)}

        if is_pmp_mode:
            ece_result = bf_metrics.expected_calibration_error(
                estimates=estimates["model_probs"],
                targets=true_models,
                model_names=self.model_names,
                **kwargs.get("ece_kwargs", {}),
            )
            metrics["ece"] = ece_result

        return metrics

    def sample(self, *args, **kwargs):
        raise NotImplementedError("ModelComparisonWorkflow does not support sampling. Use estimate() instead.")

    def log_prob(self, *args, **kwargs):
        raise NotImplementedError("ModelComparisonWorkflow does not support log_prob().")

    def ancestral_sample(self, *args, **kwargs):
        raise NotImplementedError(
            "ModelComparisonWorkflow does not support ancestral_sample(). Use estimate() instead."
        )
