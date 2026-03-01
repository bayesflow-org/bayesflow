from collections.abc import Mapping, Sequence

import numpy as np

import keras

from bayesflow.adapters import Adapter
from bayesflow.datasets import OnlineDataset
from bayesflow.networks import SummaryNetwork
from bayesflow.networks.point_inference_network import PointInferenceNetwork
from bayesflow.scores import CrossEntropyScore
from bayesflow.simulators import ModelComparisonSimulator, Simulator
from bayesflow.types import Tensor
from bayesflow.utils import filter_kwargs, logging
from bayesflow.utils.serialization import serialize, serializable

from .approximator import Approximator
from ..networks.standardization import Standardization

from ._runtime import ConditionBuilder


@serializable("bayesflow.approximators")
class ModelComparisonApproximator(Approximator):
    """
    Defines an approximator for model (simulator) comparison, where the (discrete) posterior model probabilities are
    learned with a classifier.

    Uses a :class:`~bayesflow.networks.PointInferenceNetwork` with a
    :class:`~bayesflow.scores.CrossEntropyScore` to map summary/condition inputs
    to class logits and train via categorical cross-entropy.

    Parameters
    ----------
    adapter : bf.adapters.Adapter
        Adapter for data pre-processing.
    num_models : int
        Number of models (simulators) that the approximator will compare.
    classifier_network : keras.Layer
        The network backbone (e.g., an MLP) that is used for model classification.
        Internally wrapped in a :class:`~bayesflow.networks.PointInferenceNetwork`
        with a :class:`~bayesflow.scores.CrossEntropyScore`.
        The input of the classifier network is created by concatenating ``inference_conditions``
        and (optional) output of the ``summary_network``.
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
        adapter: Adapter,
        summary_network: SummaryNetwork = None,
        standardize: str | Sequence[str] | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_models = num_models
        self.adapter = adapter

        self.inference_network = PointInferenceNetwork(
            scores={"cross_entropy": CrossEntropyScore()},
            subnet=classifier_network,
        )

        self.summary_network = summary_network
        self.condition_builder = ConditionBuilder()
        self.standardizer = Standardization(standardize)

    @classmethod
    def build_dataset(
        cls,
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
        *,
        inference_variables: Tensor,
        inference_conditions: Tensor = None,
        summary_variables: Tensor = None,
        sample_weight: Tensor = None,
        stage: str = "training",
    ) -> dict[str, Tensor]:
        """
        Computes loss and tracks metrics for the classifier and summary networks.

        This method coordinates summary metric computation (if present), combines summary outputs with
        inference conditions, computes classifier logits and cross-entropy loss via the
        :class:`~bayesflow.scores.CrossEntropyScore`, and aggregates all tracked metrics into a single
        dictionary.

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

        inference_conditions = self.standardizer.maybe_standardize(
            inference_conditions, key="inference_conditions", stage=stage
        )
        summary_variables = self.standardizer.maybe_standardize(summary_variables, key="summary_variables", stage=stage)

        summary_metrics, resolved_conditions = self.condition_builder.resolve(
            self.summary_network, inference_conditions, summary_variables, stage=stage, purpose="metrics"
        )

        inference_metrics = self.inference_network.compute_metrics(
            inference_variables, conditions=resolved_conditions, sample_weight=sample_weight, stage=stage
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
        adapter: Adapter = "auto",
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
            Additional keyword arguments passed to `keras.Model.fit()`, as described in:

        https://github.com/keras-team/keras/blob/v3.13.2/keras/src/backend/tensorflow/trainer.py#L314

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
            adapter = self.build_adapter(num_models=self.num_models, **filter_kwargs(kwargs, self.build_adapter))

        if simulator is not None:
            return super().fit(simulator=simulator, adapter=adapter, **kwargs)

        logging.info(f"Building model comparison simulator from {len(simulators)} simulators.")

        simulator = ModelComparisonSimulator(simulators=simulators)

        return super().fit(simulator=simulator, adapter=adapter, **kwargs)

    def get_config(self):
        base_config = super().get_config()

        config = {
            "num_models": self.num_models,
            "adapter": self.adapter,
            "classifier_network": self.inference_network.subnet,
            "summary_network": self.summary_network,
            "standardize": self.standardizer.standardize,
        }

        return base_config | serialize(config)

    def predict(
        self,
        *,
        conditions: Mapping[str, np.ndarray],
        probs: bool = True,
        **kwargs,
    ) -> np.ndarray:
        """
        Predicts posterior model probabilities given input conditions. The `conditions` dictionary is preprocessed
        using the `adapter`. The output is converted to NumPy array after inference.

        Parameters
        ----------
        conditions : Mapping[str, np.ndarray]
            Dictionary of conditioning variables as NumPy arrays.
        probs: bool, optional
            A flag indicating whether model probabilities (True) or logits (False) are returned. Default is True.
        **kwargs : dict
            Additional keyword arguments for the adapter and classifier.

        Returns
        -------
        outputs: np.ndarray
            Predicted posterior model probabilities given `conditions`.
        """

        # Apply adapter transforms to raw simulated / real quantities
        conditions = self.adapter(conditions, strict=False, **kwargs)

        # Ensure only keys relevant for prediction are present in the conditions dictionary
        conditions = {k: v for k, v in conditions.items() if k in ["inference_conditions", "summary_variables"]}
        conditions = keras.tree.map_structure(keras.ops.convert_to_tensor, conditions)

        # Optionally standardize conditions
        conditions["inference_conditions"] = self.standardizer.maybe_standardize(
            conditions.get("inference_conditions"), key="inference_conditions", stage="inference"
        )
        conditions["summary_variables"] = self.standardizer.maybe_standardize(
            conditions.get("summary_variables"), key="summary_variables", stage="inference"
        )

        output = self._predict(
            inference_conditions=conditions.get("inference_conditions"),
            summary_variables=conditions.get("summary_variables"),
            **kwargs,
        )

        if probs:
            output = keras.ops.softmax(output)

        return keras.ops.convert_to_numpy(output)

    def _predict(self, inference_conditions: Tensor = None, summary_variables: Tensor = None) -> Tensor:
        """Helper method to obtain logits from the classifier network."""
        if (self.summary_network is None) != (summary_variables is None):
            raise ValueError("Summary variables and summary network must be used together.")

        _, resolved_conditions = self.condition_builder.resolve(
            self.summary_network,
            inference_conditions=inference_conditions,
            summary_variables=summary_variables,
            stage="inference",
            purpose="call",
        )

        output = self.inference_network(xz=None, conditions=resolved_conditions)
        return output["cross_entropy"]["logits"]

    def _batch_size_from_data(self, data: Mapping[str, any]) -> int:
        """
        Fetches the current batch size from an input dictionary. Can only be used during training when
        inference variables (one-hot model indices) are present.
        """
        return keras.ops.shape(data["inference_variables"])[0]
