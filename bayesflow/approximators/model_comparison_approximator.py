from collections.abc import Mapping, Sequence
from typing import Literal

import numpy as np

import keras

from bayesflow.adapters import Adapter
from bayesflow.networks import ScoringRuleNetwork, SummaryNetwork
from bayesflow.scoring_rules import CategoricalScoringRule
from bayesflow.simulators import ModelComparisonSimulator, Simulator
from bayesflow.types import Tensor
from bayesflow.utils import filter_kwargs, logging
from bayesflow.utils.serialization import serializable

from .approximator import Approximator
from .scoring_rule_approximator import ScoringRuleApproximator


@serializable("bayesflow.approximators")
class ModelComparisonApproximator(ScoringRuleApproximator):
    """
    Defines an approximator for model (simulator) comparison, where the (discrete)
    posterior model probabilities are learned with a classifier implementing multi-
    class, stabilized versions of all losses described in [1].

    Uses a :class:`~bayesflow.networks.ScoringRuleNetwork` with a categorical scoring
    rule (e.g. :class:`~bayesflow.scoring_rules.CrossEntropyScore`) to map
    summary/condition inputs to class logits and train via the chosen scoring rule.
    Multiple scoring rules can be co-learned simultaneously by passing a dict of rules
    to :class:`~bayesflow.networks.ScoringRuleNetwork`. The combined loss is the mean
    of all individual scoring rule losses.

    [1] Jeffrey, N., & Wandelt, B. D. (2024). Evidence Networks: simple losses for fast,
    amortized, neural Bayesian model comparison. Machine Learning: Science and Technology,
    5(1), 015008.

    Parameters
    ----------
    inference_network : ScoringRuleNetwork
        A :class:`~bayesflow.networks.ScoringRuleNetwork` configured with one or more
        categorical scoring rules (e.g. :class:`~bayesflow.scoring_rules.CrossEntropyScore`,
        :class:`~bayesflow.scoring_rules.ExponentialScore`).
    adapter : bf.adapters.Adapter, optional
        Adapter for data pre-processing. If ``None`` (default), an identity
        adapter is used that makes a shallow copy and passes data through unchanged.
    summary_network : bf.networks.SummaryNetwork, optional
        The summary network used for data summarization (default is None).
        The input of the summary network is ``summary_variables``.
    standardize : str | Sequence[str] | None
        The variables to standardize before passing to the networks. Can be any subset
        of ["inference_conditions", "summary_variables"].
        Note: ``inference_variables`` (one-hot model indices) are never standardized,
        even when ``standardize="all"`` is passed.
        (default is None).
    """

    def __init__(
        self,
        *,
        inference_network: ScoringRuleNetwork,
        adapter: Adapter | None = None,
        summary_network: SummaryNetwork | None = None,
        standardize: str | Sequence[str] | None = "all",
        **kwargs,
    ):
        # Model indices (one-hot encoded) must never be standardized.
        standardize = self._filter_standardize(standardize)

        super().__init__(
            inference_network=inference_network,
            adapter=adapter,
            summary_network=summary_network,
            standardize=standardize,
            **kwargs,
        )

    @staticmethod
    def _filter_standardize(standardize):
        """Exclude model indices from standardization."""
        match standardize:
            case None | "inference_variables":
                return None
            case "all":
                return ["summary_variables", "inference_conditions"]
            case str():
                return standardize
            case _:
                return [item for item in standardize if item != "inference_variables"] or None

    @staticmethod
    def _rule_output_to_model_probs(
        rule: CategoricalScoringRule, rule_output: dict[str, Tensor]
    ) -> dict[str, np.ndarray]:
        """Convert a single scoring rule's raw network output to model probabilities and log odds."""
        log_odds = rule.to_log_odds(rule_output)
        model_probs = keras.ops.softmax(log_odds)
        log_odds = log_odds - log_odds[..., :1]
        return {
            "model_probs": keras.ops.convert_to_numpy(model_probs),
            "log_odds": keras.ops.convert_to_numpy(log_odds),
        }

    @staticmethod
    def _merge_rule_estimates(per_rule: dict[str, dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
        """Pool the per-rule estimates of several scoring rules into the single-rule structure.

        Each rule is a consistent estimator of the same posterior over models, so their
        ``log_odds`` are combined by a *logarithmic opinion pool* (the mean in log-odds
        space), and ``model_probs`` is derived from the pooled ``log_odds`` so the two stay
        mutually consistent.
        """
        merged_log_odds = np.mean([r["log_odds"] for r in per_rule.values()], axis=0)
        exp = np.exp(merged_log_odds - merged_log_odds.max(axis=-1, keepdims=True))
        model_probs = exp / exp.sum(axis=-1, keepdims=True)
        return {"model_probs": model_probs, "log_odds": merged_log_odds}

    def _add_log_bayes_factors(self, estimates: dict, model_prior: np.ndarray | None) -> dict:
        """Add prior-adjusted log Bayes factors to each leaf estimate dict, in place."""
        if model_prior is None:
            return estimates

        if "log_odds" in estimates:
            log_prior = np.log(model_prior)
            log_prior_odds = log_prior - log_prior[0]
            estimates["log_bayes_factors"] = estimates["log_odds"] - log_prior_odds
        else:
            for value in estimates.values():
                if isinstance(value, dict):
                    self._add_log_bayes_factors(value, model_prior)

        return estimates

    def build_dataset(
        self,
        *,
        dataset: keras.utils.PyDataset | None = None,
        simulator: ModelComparisonSimulator | None = None,
        simulators: Sequence[Simulator] | None = None,
        **kwargs,
    ) -> keras.utils.PyDataset:
        if sum(arg is not None for arg in (dataset, simulator, simulators)) != 1:
            raise ValueError("Exactly one of dataset, simulator, or simulators must be provided.")

        if simulators is not None:
            simulator = ModelComparisonSimulator(simulators)

        return super().build_dataset(dataset=dataset, simulator=simulator, **kwargs)

    def fit(
        self,
        *,
        adapter: Adapter | Literal["auto"] = "auto",
        dataset: keras.utils.PyDataset | None = None,
        simulator: ModelComparisonSimulator | None = None,
        simulators: Sequence[Simulator] | None = None,
        **kwargs,
    ):
        """
        Trains the approximator on the provided dataset or on-demand generated from the given (multi-model) simulator.
        If `dataset` is not provided, a dataset is built from the `simulator`.
        If `simulator` is not provided, it will be built from a list of `simulators`.
        If the model has not been built, it will be built using a batch from the dataset.

        Parameters
        ----------
        adapter : Adapter or 'auto', optional
            The data adapter that will make the simulated / real outputs neural-network friendly.
        dataset : keras.utils.PyDataset, optional
            A dataset containing simulations for training. If provided, `simulator` must be None.
        simulator : ModelComparisonSimulator, optional
            A simulator used to generate a dataset. If provided, `dataset` must be None.
        simulators: Sequence[Simulator], optional
            A list of simulators (one simulator per model). If provided, `dataset` must be None.
        **kwargs
            Additional keyword arguments passed to ``keras.Model.fit()``.

        Returns
        -------
        keras.callbacks.History
            A history object containing the training loss and metrics values.

        Raises
        ------
        ValueError
            If both `dataset` and `simulator` or `simulators` are provided or neither is provided.
        """
        if dataset is not None:
            if simulator is not None or simulators is not None:
                raise ValueError(
                    "Received conflicting arguments. Please provide either a dataset or a simulator, but not both."
                )
            # Let ContinuousApproximator.fit inject self.adapter
            return super().fit(dataset=dataset, **kwargs)

        if adapter == "auto":
            logging.info("Building automatic data adapter.")
            adapter = self.build_adapter(
                inference_variables="model_indices", **filter_kwargs(kwargs, self.build_adapter)
            )

        if simulator is not None:
            # Bypass ContinuousApproximator.fit to avoid duplicate adapter kwarg
            return Approximator.fit(self, simulator=simulator, adapter=adapter, **kwargs)

        logging.info(f"Building model comparison simulator from {len(simulators)} simulators.")
        simulator = ModelComparisonSimulator(simulators=simulators)
        return Approximator.fit(self, simulator=simulator, adapter=adapter, **kwargs)

    def estimate(
        self,
        *,
        conditions: Mapping[str, np.ndarray],
        return_summaries: bool = False,
        merge_scores: bool = True,
        model_prior: np.ndarray | None = None,
        **kwargs,
    ) -> dict[str, np.ndarray]:
        """
        Returns estimates given input conditions, adapting output to the active scoring rule(s).

        Every scoring rule yields ``"model_probs"`` (the posterior over ``num_models`` models)
        and length-``M`` ``"log_odds"`` relative to the reference model :math:`\\mathcal{M}_0`
        (so ``log_odds[..., 0] = 0``). When ``model_prior`` is given, ``"log_bayes_factors"``
        (same shape) is also returned, removing the training model prior:
        :math:`\\log K_{k,0} = \\text{log\\_odds}_k - (\\log \\pi_k - \\log \\pi_0)`. Without a
        prior the Bayes factors are undefined and omitted.

        Parameters
        ----------
        conditions : Mapping[str, np.ndarray]
            Dictionary of conditioning variables as NumPy arrays.
        return_summaries: bool, optional
            If set to True and a summary network is present, will return the learned summary statistics for
            the provided conditions.
        merge_scores : bool, optional
            Only relevant when more than one scoring rule is configured (default: True).
            If True, the per-rule estimates are pooled into the flat single-rule structure:
            all rules are pooled in log-odds space (logarithmic opinion pool) and a consistent
            ``"model_probs"`` is derived from the pooled ``"log_odds"``. If False, results are
            returned as a nested dict keyed by rule name. Has no effect for a single rule.
        model_prior : np.ndarray, optional
            Prior model probabilities (length ``num_models``) used during training. If given,
            the output includes ``"log_bayes_factors"``.
        **kwargs
            Additional keyword arguments forwarded to the adapter and classifier.

        Returns
        -------
        dict[str, np.ndarray]
            Single scoring rule: ``"model_probs"`` and ``"log_odds"`` of shape
            ``(num_datasets, num_models)`` (``"log_odds"`` relative to model 0, so
            ``[..., 0] = 0``), and ``"log_bayes_factors"`` of the same shape when
            ``model_prior`` is given. If a summary network is present, also contains
            ``"_summaries"``.

            Multiple scoring rules with ``merge_scores=True`` (default): the same flat
            structure, with ``"log_odds"`` pooled across rules (logarithmic opinion pool) and
            ``"model_probs"`` (and ``"log_bayes_factors"``, if a prior was given) derived from it.

            Multiple scoring rules with ``merge_scores=False``: nested dict keyed by rule
            name, each value having the same structure as the single-rule case.
            ``"_summaries"`` (if present) is at the top level in both cases.
        """
        resolved_conditions, adapted, summary_outputs = self._prepare_conditions(conditions, **kwargs)
        inference_kwargs = self._collect_mask_kwargs(self._INFERENCE_MASK_KEYS, adapted)

        raw = self.inference_network(xz=None, conditions=resolved_conditions, **inference_kwargs)

        per_rule = {
            rule_key: self._rule_output_to_model_probs(self.inference_network.scoring_rules[rule_key], rule_output)
            for rule_key, rule_output in raw.items()
        }

        if len(per_rule) == 1:
            result = next(iter(per_rule.values()))
        elif merge_scores:
            result = self._merge_rule_estimates(per_rule)
        else:
            result = per_rule

        self._add_log_bayes_factors(result, model_prior)

        if return_summaries and summary_outputs is not None:
            result["_summaries"] = keras.ops.convert_to_numpy(summary_outputs)

        return result

    def log_prob(self, *args, **kwargs):
        raise NotImplementedError("ModelComparisonApproximator does not support log_prob(). Use estimate() instead.")
