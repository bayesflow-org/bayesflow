from collections.abc import Callable, Mapping, Sequence
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import keras

from bayesflow.networks import ScoringRuleNetwork, SummaryNetwork
from bayesflow.simulators import ModelComparisonSimulator, Simulator
from bayesflow.adapters import Adapter
from bayesflow.approximators import ModelComparisonApproximator
from bayesflow.scoring_rules import CategoricalScoringRule
from bayesflow.utils import (
    find_scoring_rule,
    find_summary_network,
    filter_kwargs,
)
from bayesflow.diagnostics import plots as bf_plots
from bayesflow.diagnostics import metrics as bf_metrics

from .basic_workflow import BasicWorkflow


class ModelComparisonWorkflow(BasicWorkflow):
    """
    This class extends :class:`~bayesflow.workflows.BasicWorkflow` to support
    amortized Bayesian model comparison with scoring rules.

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
    subnet : keras.Layer or str, optional
        The shared body of the internal :class:`~bayesflow.networks.ScoringRuleNetwork`,
        on top of which one head per scoring rule is built (default: ``"mlp"``).
        Accepts a Keras layer instance or any name recognised by
        :func:`~bayesflow.utils.find_network` (e.g. ``"mlp"``).
    scoring_rules : CategoricalScoringRule or Sequence or dict[str, CategoricalScoringRule] or str, optional
        Scoring rule(s) used to train the classifier. Accepts, in increasing generality:

        - a single :class:`~bayesflow.scoring_rules.CategoricalScoringRule` instance;
        - a string name recognised by :func:`~bayesflow.utils.find_scoring_rule`;
        - a list/tuple of scores (or names) to co-learn several scores simultaneously,
          which are auto-named after their class (e.g. ``"cross_entropy_score"``);
        - a mapping of explicit names to scores.

        Multiple scores share one classifier body but get separate heads and are trained
        jointly. Their outputs can then be kept per-rule or aggregated at estimation time
        via the ``merge_scores`` argument of :meth:`estimate`. Defaults to ``"cross_entropy"``.
        The chosen rule(s) determine the training dynamics. All scores produce estimates
        for posterior model probabilities, log odds, and potentially log Bayes factors (BFs).
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
    use_mixed_batches : bool, optional
        Whether each training batch mixes samples from different models
        (default: ``True``). Forwarded to
        :class:`~bayesflow.simulators.ModelComparisonSimulator` when ``simulator``
        is a list.
    model_names : Sequence[str], optional
        Human-readable names for each model, used in diagnostic plots.
    standardize : Sequence[str] or str or None, optional
        Variables to standardize. Defaults to all available input variables.
        ``inference_variables`` (one-hot model indices) are never standardized.
    **kwargs : dict, optional
        Additional keyword arguments organised by context:

        - ``subnet_kwargs``     : dict — used in :class:`~bayesflow.networks.ScoringRuleNetwork`
          to construct ``subnet`` when it is given as a name.
        - ``summary_kwargs``    : dict — passed to :func:`~bayesflow.utils.find_summary_network`.
        - ``optimizer_kwargs``  : dict — passed to ``_init_optimizer``.
        - ``simulator_kwargs``  : dict — passed to :class:`~bayesflow.simulators.ModelComparisonSimulator`.
        - Other keys forwarded to :class:`~bayesflow.approximators.ModelComparisonApproximator`.
    """

    def __init__(
        self,
        simulator: Sequence[Simulator] | ModelComparisonSimulator | None = None,
        adapter: Adapter | None = None,
        subnet: keras.Layer | str = "mlp",
        summary_network: SummaryNetwork | str | None = None,
        scoring_rules: CategoricalScoringRule
        | Sequence[CategoricalScoringRule | str]
        | dict[str, CategoricalScoringRule]
        | str = "cross_entropy",
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
        standardize: Sequence[str] | str | None = "all",
        **kwargs,
    ):
        if isinstance(simulator, Sequence):
            simulator = ModelComparisonSimulator(
                simulators=simulator,
                shared_simulator=shared_simulator,
                use_mixed_batches=use_mixed_batches,
                **kwargs.get("simulator_kwargs", {}),
            )

        self.simulator = simulator
        self.model_names = model_names

        adapter = adapter or ModelComparisonWorkflow.default_adapter(
            inference_conditions=inference_conditions, summary_variables=summary_variables
        )

        resolved_scoring_rules = ModelComparisonWorkflow._resolve_scoring_rules(scoring_rules)

        self.approximator = ModelComparisonApproximator(
            inference_network=ScoringRuleNetwork(
                scoring_rules=resolved_scoring_rules,
                subnet=subnet,
                subnet_kwargs=kwargs.get("subnet_kwargs", {}),
            ),
            summary_network=find_summary_network(summary_network, **kwargs.get("summary_kwargs", {})),
            adapter=adapter,
            standardize=standardize,
            **filter_kwargs(kwargs, ModelComparisonApproximator.__init__),
        )

        self._init_optimizer(initial_learning_rate, optimizer, **kwargs.get("optimizer_kwargs", {}))
        self._init_checkpointing(checkpoint_filepath, checkpoint_name, save_weights_only, save_best_only)
        self.history = None
        self._needs_compile = True

    @staticmethod
    def _resolve_scoring_rules(
        scoring_rules: CategoricalScoringRule
        | Sequence[CategoricalScoringRule | str]
        | Mapping[str, CategoricalScoringRule | str]
        | str
        | None,
    ) -> dict[str, CategoricalScoringRule]:
        """Normalize the ``scoring_rules`` argument to a ``dict[str, CategoricalScoringRule]``.

        Accepts a single rule (or its string name), a list/tuple of scores (or names), or a
        mapping of names to scores. ``None`` defaults to a single ``CrossEntropyScore``.
        Scores given without an explicit name (i.e. in a list) are keyed by a snake-cased
        version of their class name, de-duplicated with a numeric suffix.

        Raises
        ------
        TypeError
            If ``scoring_rules`` is of an unsupported type.
        """

        if isinstance(scoring_rules, (str, CategoricalScoringRule)):
            return {"scoring_rule": find_scoring_rule(scoring_rules)}

        if isinstance(scoring_rules, Mapping):
            resolved = {key: find_scoring_rule(rule) for key, rule in scoring_rules.items()}

        elif isinstance(scoring_rules, Sequence):
            resolved: dict[str, CategoricalScoringRule] = {}
            counts: dict[str, int] = {}
            for item in scoring_rules:
                rule = find_scoring_rule(item)
                base = re.sub(r"(?<!^)(?=[A-Z])", "_", type(rule).__name__).lower()
                counts[base] = counts.get(base, 0) + 1
                key = base if counts[base] == 1 else f"{base}_{counts[base]}"
                resolved[key] = rule
        else:
            raise TypeError(
                "`scoring_rules` must be a CategoricalScoringRule, a string name, a list of "
                f"scores/names, or a dict of name -> rule, got {type(scoring_rules).__name__}."
            )

        return resolved

    @staticmethod
    def default_adapter(
        inference_conditions: Sequence[str] | str,
        summary_variables: Sequence[str] | str,
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

        Returns
        -------
        Adapter
            A configured Adapter instance that applies dtype conversion and concatenation.
        """
        return BasicWorkflow.default_adapter(
            inference_variables="model_indices",
            summary_variables=summary_variables,
            inference_conditions=inference_conditions,
        )

    def plot_default_diagnostics(
        self,
        test_data: Mapping[str, np.ndarray] | int,
        estimates: Mapping[str, np.ndarray] | None = None,
        true_log_bfs_fn: Callable | None = None,
        **kwargs,
    ) -> dict[str, plt.Figure]:
        r"""
        Generate default diagnostic plots for model comparison.

        Produces a loss curve (when training history is available) followed by
        a set of plots that depend on the scoring rule families present.  With
        multiple scoring rules the diagnostics run on the pooled estimates (see
        :meth:`estimate`), so all plots are produced for a single rule, multiple
        scores, and mixed families alike.

        Always produced:

        - ``"confusion_and_bayes_factors"`` — the posterior model probability
          confusion matrix (PMP) and the pairwise Bayes factor (BF) heatmap,
          drawn side by side in a single figure.
        - ``"calibration"`` — per-model calibration curves with ECE annotations.
        - ``"bayes_factor_recovery"`` — scatter of predicted vs. true log Bayes
          factors, one panel per competing model.  Only produced when
          ``true_log_bfs_fn`` is supplied.

        Parameters
        ----------
        test_data : Mapping[str, np.ndarray] or int
            Either a pre-simulated data dictionary (as returned by
            :meth:`simulate`) or an integer specifying how many datasets to
            generate using the attached simulator.
        estimates: Mapping[str, np.ndarray] or None
            Optional pre-computed dictionary of estimates corresponding to the quantities
            in ``test_data``. Default is ``None`` (i.e., esimtates are obtained from the
            test data by calling the workflow's ``.estimate()`` method).
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
            - ``matrix_row_figsize`` — figure size ``(width, height)`` for the combined
              side-by-side confusion matrix / pairwise Bayes factor figure, used only
              when both scoring rule families are present. Inferred from the number of
              models if omitted.

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

        if estimates is None:
            # Diagnostics operate on merged 'model_probs'; force it regardless of any user estimate_kwargs.
            estimate_kwargs = kwargs.get("estimate_kwargs", {})
            estimate_kwargs["merge_scores"] = True
            estimates = self.estimate(conditions=test_data, **estimate_kwargs)

        # Always draw the confusion matrix (PMP view) and pairwise Bayes factor heatmap
        model_probs = estimates["model_probs"]
        pred_log_bfs = estimates.get("log_bayes_factors", estimates["log_odds"])[..., 1:]

        num_models = model_probs.shape[-1]
        size = max(4.0, num_models * 1.2)
        matrix_fig, matrix_axes = plt.subplots(1, 2, figsize=kwargs.get("matrix_row_figsize", (2 * size + 1.0, size)))
        # Smaller fonts so two titled panels read as one row without a grand title;
        # user-supplied kwargs still win.
        panel_defaults = dict(title_fontsize=14, label_fontsize=13, tick_fontsize=11, value_fontsize=9)

        bf_plots.mc_confusion_matrix(
            pred_models=model_probs,
            true_models=true_models,
            model_names=self.model_names,
            ax=matrix_axes[0],
            **{**panel_defaults, **kwargs.get("confusion_matrix_kwargs", {})},
        )
        bf_plots.pairwise_bayes_factors(
            pred_log_bayes_factors=pred_log_bfs,
            true_models=true_models,
            model_names=self.model_names,
            ax=matrix_axes[1],
            **{**panel_defaults, **kwargs.get("pairwise_bayes_factors_kwargs", {})},
        )
        matrix_fig.tight_layout()
        figures["confusion_and_bayes_factors"] = matrix_fig

        figures["calibration"] = bf_plots.mc_calibration(
            pred_models=estimates["model_probs"],
            true_models=true_models,
            model_names=self.model_names,
            **kwargs.get("calibration_kwargs", {}),
        )

        if true_log_bfs_fn is not None:
            true_log_bfs = true_log_bfs_fn(test_data)
            figures["bayes_factor_recovery"] = bf_plots.bayes_factor_recovery(
                pred_log_bayes_factors=estimates.get("log_bayes_factors", estimates["log_odds"])[..., 1:],
                true_log_bayes_factors=true_log_bfs,
                true_models=true_models,
                model_names=self.model_names,
                **kwargs.get("bayes_factor_recovery_kwargs", {}),
            )

        return figures

    def compute_default_diagnostics(
        self,
        test_data: Mapping[str, np.ndarray] | int,
        estimates: Mapping[str, np.ndarray] | None = None,
        as_data_frame: bool = True,
        **kwargs,
    ) -> Sequence[dict] | pd.DataFrame:
        """
        Compute default scalar diagnostic metrics for model comparison.

        All metrics are computed from the pooled log odds, see :meth:`estimate`):

        - ``"accuracy"`` — per-model classification accuracy, i.e., the diagonal
          entries of the row-normalized confusion matrix (fraction of datasets
          simulated from model ``m`` for which ``argmax(PMPs)`` also predicts
          model ``m``), as returned by
          :func:`~bayesflow.diagnostics.metrics.accuracy`. Controlled
          via the ``per_model`` keyword (default ``True``); set to ``False`` to
          instead obtain the overall (aggregate) accuracy as a single float.
        - ``"ece"`` — per-model expected calibration errors, as returned by
          :func:`~bayesflow.diagnostics.metrics.expected_calibration_error`.
        - ``"brier_score"`` — per-model (one-vs-rest) and aggregate multi-class
          Brier scores, as returned by
          :func:`~bayesflow.diagnostics.metrics.brier_score`.

        Parameters
        ----------
        test_data : Mapping[str, np.ndarray] or int
            Either a pre-simulated data dictionary or an integer specifying how many
            datasets to generate using the attached simulator.
        estimates: Mapping[str, np.ndarray] or None
            Optional pre-computed dictionary of estimates corresponding to the quantities
            in ``test_data``. Default is ``None`` (i.e., esimtates are obtained from the
            test data by calling the workflow's ``.estimate()`` method).
        as_data_frame : bool, optional
            Whether to return the results as a pandas DataFrame (default: True),
            with one column per model and one row per metric. If False, a
            dictionary of the raw (unparsed) metric outputs is returned instead.
        **kwargs : dict, optional
            Fine-grained control via nested dicts:

            - ``test_data_kwargs`` — forwarded to :meth:`simulate` when ``test_data``
              is an integer.
            - ``estimate_kwargs`` — forwarded to :meth:`estimate`.
            - ``accuracy_kwargs`` — forwarded to
              :func:`~bayesflow.diagnostics.metrics.accuracy`
              (e.g. ``{"per_model": False}`` for the aggregate accuracy).
            - ``ece_kwargs`` — forwarded to
              :func:`~bayesflow.diagnostics.metrics.expected_calibration_error`.

        Returns
        -------
        dict[str, Any] or pd.DataFrame
            If ``as_data_frame`` is True, a pandas DataFrame with one row per
            metric and one column per model. Otherwise, a dictionary of the raw
            metric outputs.

        Raises
        ------
        ValueError
            If ``test_data`` is an integer but no simulator is attached.
        """
        if isinstance(test_data, int):
            if self.simulator is None:
                raise ValueError(f"No simulator attached. Cannot generate {test_data} test datasets.")
            test_data = self.simulate(test_data, **kwargs.get("test_data_kwargs", {}))

        if estimates is None:
            # Diagnostics operate on merged 'model_probs'; force it regardless of any user estimate_kwargs.
            estimate_kwargs = kwargs.get("estimate_kwargs", {})
            estimate_kwargs["merge_scores"] = True
            estimates = self.estimate(conditions=test_data, **estimate_kwargs)

        if "model_indices" in test_data:
            true_models = test_data["model_indices"]
        elif "inference_variables" in test_data:
            true_models = test_data["inference_variables"]
        else:
            raise KeyError(
                "test_data must contain 'model_indices' (raw simulator output) or "
                "'inference_variables' (adapted output). Neither key was found."
            )

        accuracy_kwargs = dict(kwargs.get("accuracy_kwargs", {}))
        accuracy_kwargs.setdefault("per_model", True)

        # Merged estimates always carry 'model_probs' (derived from the pooled log Bayes
        # factors for BF rules), so all metrics apply to both rule families.
        accuracy_result = bf_metrics.accuracy(
            estimates["model_probs"],
            true_models,
            model_names=self.model_names,
            **accuracy_kwargs,
        )
        ece = bf_metrics.expected_calibration_error(
            estimates=estimates["model_probs"],
            targets=true_models,
            model_names=self.model_names,
            **kwargs.get("ece_kwargs", {}),
        )
        brier_score_result = bf_metrics.brier_score(
            estimates=estimates["model_probs"],
            targets=true_models,
            model_names=self.model_names,
        )

        if as_data_frame:
            model_names = ece["model_names"]
            if isinstance(accuracy_result, dict):
                accuracy_values = accuracy_result["values"]
            else:
                # Aggregate accuracy (per_model=False): broadcast the single value to all models.
                accuracy_values = [accuracy_result] * len(model_names)

            metrics = pd.DataFrame(
                {
                    "Accuracy": accuracy_values,
                    ece["metric_name"]: ece["values"],
                    brier_score_result["metric_name"]: brier_score_result["values"],
                },
                index=model_names,
            ).T
        else:
            metrics = {"accuracy": accuracy_result, "ece": ece, "brier_score": brier_score_result}

        return metrics

    def sample(self, *args, **kwargs):
        raise NotImplementedError("ModelComparisonWorkflow does not support sampling. Use estimate() instead.")

    def log_prob(self, *args, **kwargs):
        raise NotImplementedError("ModelComparisonWorkflow does not support log_prob().")

    def ancestral_sample(self, *args, **kwargs):
        raise NotImplementedError(
            "ModelComparisonWorkflow does not support ancestral_sample(). Use estimate() instead."
        )
