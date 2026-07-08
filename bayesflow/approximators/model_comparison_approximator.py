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
        standardize: str | Sequence[str] | None = None,
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
        """Convert a single scoring rule's raw network output to model probabilities."""
        if "logits" in rule_output:
            logits = rule_output["logits"]
            return {
                "logits": keras.ops.convert_to_numpy(logits),
                "model_probs": keras.ops.convert_to_numpy(keras.ops.softmax(logits)),
            }

        # Bayes factor mode
        log_bfs = rule.to_bayes_factors(rule_output["log_bayes_factors"])
        f0 = keras.ops.zeros_like(log_bfs[..., :1])
        model_probs = keras.ops.softmax(keras.ops.concatenate([f0, log_bfs], axis=-1))

        return {
            "log_bayes_factors": keras.ops.convert_to_numpy(log_bfs),
            "model_probs": keras.ops.convert_to_numpy(model_probs),
        }

    @staticmethod
    def _merge_rule_estimates(per_rule: dict[str, dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
        """Aggregate the per-rule estimates of several scoring rules into the flat single-rule structure.

        Each rule is a consistent estimator of the same posterior over models, so the
        estimates are pooled across rules:

        - **Any Bayes factor rule present**: a *logarithmic opinion pool* in log-odds space.
          BF rules contribute their ``log_bayes_factors`` (already on a common scale via
          ``to_bayes_factors``); PMP rules contribute the pairwise differences of their
          ``logits``, which equal the same log posterior odds exactly (softmax is invariant
          to the per-row shift). The averages are returned as ``log_bayes_factors``, and
          ``model_probs`` is derived from them so the two stay mutually consistent.
        - **PMP rules only**: a *linear opinion pool* — the per-rule ``model_probs`` are
          averaged. The pre-softmax ``logits`` have no coherent pooled counterpart and are
          dropped (with a warning).
        """
        rule_results = list(per_rule.values())

        if any("log_bayes_factors" in r for r in rule_results):
            merged_log_bfs = np.mean(
                [
                    r["log_bayes_factors"] if "log_bayes_factors" in r else r["logits"][..., 1:] - r["logits"][..., :1]
                    for r in rule_results
                ],
                axis=0,
            )
            f0 = np.zeros_like(merged_log_bfs[..., :1])
            logits = np.concatenate([f0, merged_log_bfs], axis=-1)
            logits = logits - logits.max(axis=-1, keepdims=True)
            exp = np.exp(logits)
            model_probs = exp / exp.sum(axis=-1, keepdims=True)
            return {"log_bayes_factors": merged_log_bfs, "model_probs": model_probs}

        logging.warning(
            f"merge_scores=True: averaging 'model_probs' across {len(per_rule)} PMP scoring rules; "
            "the per-rule 'logits' are dropped. Pass merge_scores=False to retain the per-rule outputs."
        )
        model_probs = np.mean([r["model_probs"] for r in rule_results], axis=0)
        return {"model_probs": model_probs}

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
        dataset: keras.utils.PyDataset = None,
        simulator: ModelComparisonSimulator = None,
        simulators: Sequence[Simulator] = None,
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
        **kwargs,
    ) -> dict[str, np.ndarray]:
        """
        Returns estimates given input conditions, adapting output to the active scoring rule(s).

        For PMP scoring rules (e.g. :class:`~bayesflow.scoring_rules.CrossEntropyScore`,
        :class:`~bayesflow.scoring_rules.BrierScore`) the network outputs logits over
        ``num_models`` classes, converted to softmax probabilities.

        For Bayes factor scoring rules (e.g. :class:`~bayesflow.scoring_rules.ExponentialScore`)
        the network outputs ``num_models - 1`` log Bayes factors
        :math:`\\log K_{k,0} = \\log p(x \\mid \\mathcal{M}_k) - \\log p(x \\mid \\mathcal{M}_0)`
        relative to the reference model :math:`\\mathcal{M}_0`, which are converted to posterior
        model probabilities by prepending the reference anchor :math:`f_0 = 0` and applying
        softmax (assumes equal model priors).

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
            when at least one Bayes factor rule is present, all rules are pooled in log-odds
            space (logarithmic opinion pool; PMP rules enter via their logit differences) and
            a consistent ``"model_probs"`` is derived from the pooled ``"log_bayes_factors"``,
            keeping both keys; PMP-only rules average their ``"model_probs"`` (linear opinion
            pool) and drop the pre-softmax ``"logits"`` (with a warning). If False, results
            are returned as a nested dict keyed by rule name. Has no effect for a single rule.
        **kwargs
            Additional keyword arguments forwarded to the adapter and classifier.

        Returns
        -------
        dict[str, np.ndarray]
            Single scoring rule: always contains ``"model_probs"`` of shape
            ``(num_datasets, num_models)``. PMP rules additionally contain ``"logits"``;
            Bayes factor rules additionally contain ``"log_bayes_factors"``.
            If a summary network is present, also contains ``"_summaries"``.

            Multiple scoring rules with ``merge_scores=True`` (default): the flat single-rule
            structure. PMP-only rules return pooled ``"model_probs"`` only; if any Bayes factor
            rule is present, the result contains pooled ``"log_bayes_factors"`` together with
            the ``"model_probs"`` derived from them.

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

        # Single rule → flat dict (merge_scores is a no-op here).
        if len(per_rule) == 1:
            result = next(iter(per_rule.values()))
        elif merge_scores:
            result = self._merge_rule_estimates(per_rule)
        else:
            result = per_rule

        if return_summaries and summary_outputs is not None:
            result["_summaries"] = keras.ops.convert_to_numpy(summary_outputs)

        return result

    def log_prob(self, *args, **kwargs):
        raise NotImplementedError("ModelComparisonApproximator does not support log_prob(). Use estimate() instead.")
