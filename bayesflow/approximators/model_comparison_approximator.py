from collections.abc import Mapping, Sequence
from typing import Literal

import numpy as np

import keras

from bayesflow.adapters import Adapter
from bayesflow.datasets import OnlineDataset
from bayesflow.networks import ScoringRuleNetwork, SummaryNetwork
from bayesflow.scoring_rules import CrossEntropyScore, ScoringRule
from bayesflow.simulators import ModelComparisonSimulator, Simulator
from bayesflow.types import Tensor
from bayesflow.utils import filter_kwargs, logging
from bayesflow.utils.serialization import serialize, serializable

from .approximator import Approximator
from .helpers import ConditionBuilder

from ..networks.helpers import Standardization


@serializable("bayesflow.approximators")
class ModelComparisonApproximator(Approximator):
    """
    Defines an approximator for model (simulator) comparison, where the (discrete)
    posterior model probabilities are learned with a classifier.

    Per default, it uses a :class:`~bayesflow.networks.ScoringRuleNetwork` with a
    :class:`~bayesflow.scoring_rules.CrossEntropyScore` to map summary/condition inputs
    to class logits and train via categorical cross-entropy. However, you can use any
    of the losses defined in [1] and generalized to K-model scenarios.


    [1] Jeffrey, N., & Wandelt, B. D. (2024). Evidence Networks: simple losses for fast,
    amortized, neural Bayesian model comparison. Machine Learning: Science and Technology,
    5(1), 015008.

    Parameters
    ----------
    num_models : int
        Number of models (simulators) that the approximator will compare.
    classifier_network : keras.Layer
        The network backbone (e.g., an MLP) that is used for model classification.
        Internally wrapped in a :class:`~bayesflow.networks.ScoringRuleNetwork`
        with a :class:`~bayesflow.scoring_rules.CrossEntropyScore`.
        The input to the classifier network is created by concatenating ``inference_conditions``
        and (optional) output of the ``summary_network``.
    scoring_rule : bf.scoring_rules.ScoringRule, optional
        The scoring rule used for training. If ``None`` (default), a
        :class:`~bayesflow.scoring_rules.CrossEntropyScore` is used.
    adapter : bf.adapters.Adapter, optional
        Adapter for data pre-processing. If ``None`` (default), an identity
        adapter is used that makes a shallow copy and passes data through unchanged.
    summary_network : bf.networks.SummaryNetwork, optional
        The summary network used for data summarization (default is None).
        The input of the summary network is ``summary_variables``.
    standardize : str | Sequence[str] | None
        The variables to standardize before passing to the networks. Can be any subset
        of ["inference_conditions", "summary_variables"].
        (default is None, since model indices are one-hot encoded and should not be standardized).
    """

    def __init__(
        self,
        *,
        num_models: int,
        classifier_network: keras.Layer,
        scoring_rule: ScoringRule = None,
        adapter: Adapter = None,
        summary_network: SummaryNetwork = None,
        standardize: str | Sequence[str] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_models = num_models
        self.adapter = adapter if adapter is not None else Adapter()

        scoring_rule = scoring_rule if scoring_rule is not None else CrossEntropyScore()
        self.scoring_rule = scoring_rule

        self.inference_network = ScoringRuleNetwork(
            scoring_rules={"scoring_rule": scoring_rule}, subnet=classifier_network
        )

        self.summary_network = summary_network
        self.condition_builder = ConditionBuilder()
        self.standardizer = Standardization(standardize)

    def build_dataset(
        self,
        *,
        dataset: keras.utils.PyDataset = None,
        simulator: ModelComparisonSimulator = None,
        simulators: Sequence[Simulator] = None,
        **kwargs,
    ) -> OnlineDataset:
        if sum(arg is not None for arg in (dataset, simulator, simulators)) != 1:
            raise ValueError("Exactly one of dataset, simulator, or simulators must be provided.")

        if simulators is not None:
            simulator = ModelComparisonSimulator(simulators)

        return super().build_dataset(dataset=dataset, simulator=simulator, **kwargs)

    def compute_metrics(
        self,
        inference_variables: Tensor,
        inference_conditions: Tensor = None,
        summary_variables: Tensor = None,
        sample_weight: Tensor = None,
        summary_attention_mask: Tensor = None,
        summary_mask: Tensor = None,
        inference_attention_mask: Tensor = None,
        inference_mask: Tensor = None,
        stage: str = "training",
    ) -> dict[str, Tensor]:
        """
        Computes loss and tracks metrics for the classifier and summary networks.

        This method coordinates summary metric computation (if present), combines
        summary outputs with inference conditions, computes classifier logits and
        cross-entropy loss via the :class:`~bayesflow.scoring_rules.CrossEntropyScore`,
        and aggregates all tracked metrics into a single dictionary.

        Parameters
        ----------
        inference_variables : Tensor
            One-hot encoded model indices (targets for classification).
        inference_conditions : Tensor, optional
            Conditioning variables for the classifier network (default is None). May be
            combined with summary network outputs if present.
        summary_variables : Tensor, optional
            Input tensor(s) for the summary network (default is None). Required if a summary
            network is present.
        sample_weight : Tensor, optional
            Weighting tensor for metric computation (default is None).
        summary_attention_mask : Tensor, optional
            Attention mask forwarded to the summary network (default is None).
        summary_mask : Tensor, optional
            Padding / key mask forwarded to the summary network (default is None).
        inference_attention_mask : Tensor, optional
            Accepted for API consistency but unused (model comparison uses an MLP classifier).
        inference_mask : Tensor, optional
            Padding / key mask forwarded to the classifier network (default is None).
        stage : str, optional
            Current training stage (e.g., "training", "validation", "inference"). Controls
            certain metric computations (default is "training").

        Returns
        -------
        metrics : dict[str, Tensor]
            Dictionary containing the total loss under the key "loss", as well as all tracked
            metrics for the classifier and summary networks. Each metric key is prefixed to
            indicate its source.
        """

        masks = dict(
            summary_attention_mask=summary_attention_mask,
            summary_mask=summary_mask,
            inference_attention_mask=inference_attention_mask,
            inference_mask=inference_mask,
        )

        summary_kwargs = self._collect_mask_kwargs(self._SUMMARY_MASK_KEYS, masks)
        inference_kwargs = self._collect_mask_kwargs(self._INFERENCE_MASK_KEYS, masks)

        resolved_conditions, summary_metrics = self._standardize_and_resolve(
            inference_conditions, summary_variables, stage=stage, purpose="metrics", **summary_kwargs
        )

        inference_metrics = self.inference_network.compute_metrics(
            inference_variables,
            conditions=resolved_conditions,
            sample_weight=sample_weight,
            stage=stage,
            **inference_kwargs,
        )

        if "loss" in summary_metrics:
            loss = inference_metrics["loss"] + summary_metrics["loss"]
        else:
            loss = inference_metrics.pop("loss")

        inference_metrics = {f"{key}/inference_{key}": value for key, value in inference_metrics.items()}
        summary_metrics = {f"{key}/summary_{key}": value for key, value in summary_metrics.items()}

        metrics = {"loss": loss} | inference_metrics | summary_metrics
        return metrics

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

            return super().fit(dataset=dataset, **kwargs)

        if adapter == "auto":
            logging.info("Building automatic data adapter.")
            adapter = self.build_adapter(
                inference_variables="model_indices", **filter_kwargs(kwargs, self.build_adapter)
            )

        if simulator is not None:
            return super().fit(simulator=simulator, adapter=adapter, **kwargs)

        logging.info(f"Building model comparison simulator from {len(simulators)} simulators.")

        simulator = ModelComparisonSimulator(simulators=simulators)

        return super().fit(simulator=simulator, adapter=adapter, **kwargs)

    def get_config(self):
        base_config = super().get_config()

        config = {
            "num_models": self.num_models,
            "scoring_rule": self.scoring_rule,
            "adapter": self.adapter,
            "classifier_network": self.inference_network.subnet,
            "summary_network": self.summary_network,
            "standardize": self.standardizer.standardize,
        }

        return base_config | serialize(config)

    def estimate(
        self,
        *,
        conditions: Mapping[str, np.ndarray],
        **kwargs,
    ) -> dict[str, np.ndarray]:
        """
        Returns estimates given input conditions, adapting output to the active scoring rule.

        For PMP scoring rules (e.g. :class:`~bayesflow.scoring_rules.CrossEntropyScore`,
        :class:`~bayesflow.scoring_rules.BrierScore`) the network outputs logits over
        ``num_models`` classes, converted to softmax probabilities.

        For Bayes factor scoring rules (e.g. :class:`~bayesflow.scoring_rules.ExponentialScore`)
        the network outputs ``num_models - 1`` log Bayes factors
        :math:`\log K_{k,0} = \log p(x \mid \mathcal{M}_k) - \log p(x \mid \mathcal{M}_0)`
        relative to the reference model :math:`\mathcal{M}_0`, which are converted to posterior
        model probabilities by prepending the reference anchor :math:`f_0 = 0` and applying
        softmax (assumes equal model priors).

        Parameters
        ----------
        conditions : Mapping[str, np.ndarray]
            Dictionary of conditioning variables as NumPy arrays.
        **kwargs
            Additional keyword arguments forwarded to the adapter and classifier.

        Returns
        -------
        dict[str, np.ndarray]
            Always contains ``"model_probs"`` of shape ``(num_datasets, num_models)``.
            PMP rules additionally contain ``"logits"`` of shape ``(num_datasets, num_models)``.
            Bayes factor rules additionally contain ``"log_bayes_factors"`` of shape
            ``(num_datasets, num_models - 1)``.
        """
        resolved_conditions, adapted, _ = self._prepare_conditions(conditions, **kwargs)
        inference_kwargs = self._collect_mask_kwargs(self._INFERENCE_MASK_KEYS, adapted)

        output = self.inference_network(xz=None, conditions=resolved_conditions, **inference_kwargs)
        raw = output["scoring_rule"]

        if "logits" in raw:
            logits = raw["logits"]
            return {
                "logits": keras.ops.convert_to_numpy(logits),
                "model_probs": keras.ops.convert_to_numpy(keras.ops.softmax(logits)),
            }
        elif "log_bayes_factors" in raw:
            log_bfs = self.scoring_rule.to_bayes_factors(raw["log_bayes_factors"])
            f0 = keras.ops.zeros_like(log_bfs[..., :1])
            model_probs = keras.ops.softmax(keras.ops.concatenate([f0, log_bfs], axis=-1))
            return {
                "log_bayes_factors": keras.ops.convert_to_numpy(log_bfs),
                "model_probs": keras.ops.convert_to_numpy(model_probs),
            }
        else:
            raise RuntimeError(
                f"Unrecognized scoring rule output keys: {list(raw.keys())}. "
                "Expected 'logits' (PMP rules) or 'log_bayes_factors' (Bayes factor rules)."
            )
