from collections.abc import Mapping, Sequence

import numpy as np
from scipy.special import logsumexp
import keras

from bayesflow.adapters import Adapter
from bayesflow.simulators import Simulator
from bayesflow.types import Tensor
from bayesflow.utils.serialization import deserialize, serializable, serialize

from .approximator import Approximator


@serializable("bayesflow.approximators")
class EnsembleApproximator(Approximator):
    def __init__(self, approximators: dict[str, Approximator], **kwargs):
        super().__init__(**kwargs)

        self.approximators = approximators

    @property
    def adapter(self) -> Adapter:
        # Defer to any adapter of the approximators,
        # assuming all are the same, which is not enforced at the moment.
        # self.adapter will only be used when super().fit calls build_dataset(..., adapter=self.adapter).
        # The attribute would not be necessary if the dataset wouldn't need an adapter.
        return next(iter(self.approximators.values())).adapter
        # TODO: enforce identical adapters

    @classmethod
    def build_dataset(
        cls,
        *,
        batch_size: int = "auto",
        num_batches: int,
        adapter: Adapter = "auto",
        memory_budget: str | int = "auto",
        simulator: Simulator,
        workers: int = "auto",
        use_multiprocessing: bool = False,
        max_queue_size: int = 32,
        **kwargs,
    ):
        raise NotImplementedError("Automatic construction of an EnsembleDataset from a simulator is not yet supported.")

    # the lines below WOULD take care of automatic EnsembleDataset construction, were this not a class method.
    # len(self.approximators) cannot be called since self is not availalbe.
    # ) -> EnsembleDataset:
    #     # Build the underlying OnlineDataset using the base implementation.
    #     base_ds = super().build_dataset(
    #         batch_size=batch_size,
    #         num_batches=num_batches,
    #         adapter=adapter,
    #         memory_budget=memory_budget,
    #         simulator=simulator,
    #         workers=workers,
    #         use_multiprocessing=use_multiprocessing,
    #         max_queue_size=max_queue_size,
    #         **kwargs,
    #     )
    #
    #     # Wrap it into an EnsembleDataset
    #     return EnsembleDataset(base_ds, num_ensemble=len(self.approximators), **kwargs)

    def build_from_data(self, adapted_data: dict[str, any]):
        data_shapes = keras.tree.map_structure(keras.ops.shape, adapted_data)
        # Remove the ensemble dimension from data_shapes. This expects data_shapes are the shapes of a
        # batch of training data, where the second axis corresponds to different approximators.
        data_shapes = keras.tree.map_shape_structure(lambda shape: shape[:1] + shape[2:], data_shapes)
        self.build(data_shapes)

    def build(self, input_shape: dict[str, tuple[int] | dict[str, dict]]) -> None:
        for approximator in self.approximators.values():
            approximator.build(input_shape)

        self.distribution_keys = [k for k, approx in self.approximators.items() if approx.has_distribution]
        # Update attribute to mark whether it has at least one score that represents a distribution
        self.has_distribution = len(self.distribution_keys) > 0

    def fit(self, *args, **kwargs):
        """
        Trains the ensemble of approximators on the provided dataset or on-demand data generated
        from the given simulator.
        If `dataset` is not provided, a dataset is built from the `simulator`.
        If the model has not been built, it will be built using a batch from the dataset.

        If `dataset` is `OnlineDataset`, `OfflineDataset` or `DiskDataset`,
        it will be wrapped into an `EnsembleDataset`.

        Parameters
        ----------
        dataset : keras.utils.PyDataset, optional
            A dataset containing simulations for training. If provided, `simulator` must be None.
        simulator : Simulator, optional
            A simulator used to generate a dataset. If provided, `dataset` must be None.
            NOTE: CURRENTLY PASSING A simulator DIRECTLY TO fit IS NOT SUPPORTED.
            PASS A EnsembleDataset instead.
        **kwargs
            Additional keyword arguments passed to `keras.Model.fit()`, including (see also `build_dataset`):

            batch_size : int or None, default='auto'
                Number of samples per gradient update. Do not specify if `dataset` is provided as a
                `keras.utils.PyDataset`, `tf.data.Dataset`, `torch.utils.data.DataLoader`, or a generator function.
            epochs : int, default=1
                Number of epochs to train the model.
            verbose : {"auto", 0, 1, 2}, default="auto"
                Verbosity mode. 0 = silent, 1 = progress bar, 2 = one line per epoch.
            callbacks : list of keras.callbacks.Callback, optional
                List of callbacks to apply during training.
            validation_split : float, optional
                Fraction of training data to use for validation (only supported if `dataset` consists of NumPy arrays
                or tensors).
            validation_data : tuple or dataset, optional
                Data for validation, overriding `validation_split`.
            shuffle : bool, default=True
                Whether to shuffle the training data before each epoch (ignored for dataset generators).
            initial_epoch : int, default=0
                Epoch at which to start training (useful for resuming training).
            steps_per_epoch : int or None, optional
                Number of steps (batches) before declaring an epoch finished.
            validation_steps : int or None, optional
                Number of validation steps per validation epoch.
            validation_batch_size : int or None, optional
                Number of samples per validation batch (defaults to `batch_size`).
            validation_freq : int, default=1
                Specifies how many training epochs to run before performing validation.

        Returns
        -------
        keras.callbacks.History
            A history object containing the training loss and metrics values.

        Raises
        ------
        ValueError
            If both `dataset` and `simulator` are provided or neither is provided.
        """
        return super().fit(*args, **kwargs, adapter=self.adapter)

    def compute_metrics(
        self,
        inference_variables: Tensor,
        inference_conditions: Tensor = None,
        summary_variables: Tensor = None,
        sample_weight: Tensor = None,
        stage: str = "training",
    ) -> dict[str, dict[str, Tensor]]:
        # Prepare empty dict for metrics
        metrics = {}

        # Define the variable slices as None (default) or respective input
        _inference_variables = inference_variables
        _inference_conditions = inference_conditions
        _summary_variables = summary_variables
        _sample_weight = sample_weight

        for i, (approx_name, approximator) in enumerate(self.approximators.items()):
            # During training each approximator receives its own separate slice from an EnsembleDataset
            if stage == "training":
                # Pick out the correct slice for each ensemble member
                _inference_variables = inference_variables[:, i]
                if inference_conditions is not None:
                    _inference_conditions = inference_conditions[:, i]
                if summary_variables is not None:
                    _summary_variables = keras.tree.map_structure(lambda v: v[:, i], summary_variables)
                if sample_weight is not None:
                    _sample_weight = sample_weight[:, i]

            metrics[approx_name] = approximator.compute_metrics(
                inference_variables=_inference_variables,
                inference_conditions=_inference_conditions,
                summary_variables=_summary_variables,
                sample_weight=_sample_weight,
                stage=stage,
            )

        # Flatten metrics dict
        joint_metrics = {}
        for approx_name in metrics.keys():
            for metric_key, value in metrics[approx_name].items():
                joint_metrics[f"{approx_name}/{metric_key}"] = value
        metrics = joint_metrics

        # Sum over losses
        losses = [v for k, v in metrics.items() if "loss" in k]
        metrics["loss"] = keras.ops.sum(losses)

        return metrics

    def sample_separate(
        self,
        *,
        num_samples: int | Mapping[str, int],
        conditions: Mapping[str, np.ndarray],
        split: bool = False,
        **kwargs,
    ) -> dict[str, dict[str, np.ndarray]]:
        """
        Draw samples from each approximator separately.

        Parameters
        ----------
        num_samples : int or Mapping[str, int]
            Number of samples to draw from each approximator. If int, all approximators
            draw the same number of samples. If dict, specifies samples per approximator.
        conditions : Mapping[str, np.ndarray]
            Conditions for sampling.
        split : bool, optional
            Whether to split output arrays, by default False.
        **kwargs
            Additional arguments passed to approximator.sample().

        Returns
        -------
        dict[str, dict[str, np.ndarray]]
            Samples keyed by approximator name, then by variable name.
        """
        samples = {}
        # if num_samples is int, sample that many for each distribution approximator.
        if isinstance(num_samples, int):
            num_samples: np.ndarray = num_samples * np.ones(len(self.distribution_keys), dtype="int64")
            num_samples: dict[str, int] = {k: num_samples[i] for i, k in enumerate(self.distribution_keys)}
        for approx_name in num_samples.keys():
            if num_samples[approx_name] < 1:
                samples[approx_name] = None
            else:
                samples[approx_name] = self.approximators[approx_name].sample(
                    num_samples=num_samples[approx_name], conditions=conditions, split=split, **kwargs
                )
        return samples

    def sample(
        self,
        *,
        num_samples: int,
        conditions: Mapping[str, np.ndarray],
        split: bool = False,
        member_weights: Mapping[str, float] | None = None,
        **kwargs,
    ) -> dict[str, np.ndarray]:
        """
        Draw samples from the marginalized distribution over ensemble members.

        Samples are allocated to approximators via multinomial sampling using member_weights,
        then concatenated and shuffled to produce the marginal distribution.

        Parameters
        ----------
        num_samples : int
            Total number of samples to draw.
        conditions : Mapping[str, np.ndarray]
            Conditions for sampling.
        split : bool, optional
            Whether to split output arrays, by default False.
        member_weights : Mapping[str, float] or None, optional
            Probability weights for each approximator. If None, uses uniform weights.
            Must be positive, will be normalized to sum to 1.
        **kwargs
            Additional arguments passed to approximator.sample().

        Returns
        -------
        dict[str, np.ndarray]
            Samples with shape (batch_size, num_samples, ...) for each variable.
        """
        member_weights: Mapping[str, float] = self._resolve_member_weights(member_weights)
        # Sample members from multinomial and convert to dict
        num_samples_per_member = np.random.multinomial(num_samples, list(member_weights.values()))
        num_samples_per_member = {k: num_samples_per_member[i] for i, k in enumerate(member_weights.keys())}
        samples = self.sample_separate(num_samples=num_samples_per_member, conditions=conditions, split=split, **kwargs)

        # Concatenate samples from all approximators along sample dimension (axis 1)
        samples_list = [
            samples[approx_name] for approx_name in self.distribution_keys if samples[approx_name] is not None
        ]
        concatenated = keras.tree.map_structure(
            lambda *arrays: np.concatenate(arrays, axis=1),  # zip & concat across approximators
            *samples_list,  # unpack: apply lambda to corresponding leaves from each dict
        )

        # Shuffle along sample dimension (axis 1)
        shuffle_idx = np.random.permutation(num_samples)
        shuffled = keras.tree.map_structure(lambda arr: arr[:, shuffle_idx], concatenated)
        return shuffled

    def log_prob_separate(
        self, data: Mapping[str, np.ndarray], members: Sequence[str] | None = None, **kwargs
    ) -> dict[str, np.ndarray]:
        """
        Compute log probabilities from each approximator separately.

        Parameters
        ----------
        data : Mapping[str, np.ndarray]
            Data containing inference variables and conditions.
        members: Sequence[str] or None, default None
            Ensemble members to evaluate log prob for.
            If None, will evaluate all distribution members.
        **kwargs
            Additional arguments passed to approximator.log_prob().

        Returns
        -------
        dict[str, np.ndarray]
            Log probabilities keyed by approximator name, each with shape (batch_size,).
        """
        log_prob = {}
        members = self.distribution_keys if members is None else members
        for approx_name, approximator in self.approximators.items():
            if approx_name in members:
                log_prob[approx_name] = approximator.log_prob(data=data, **kwargs)
        return log_prob

    def log_prob(
        self, data: Mapping[str, np.ndarray], member_weights: Mapping[str, float] | None = None, **kwargs
    ) -> np.ndarray:
        """
        Compute the marginalized log probability over ensemble members.

        Uses log-sum-exp trick to compute log p(x) = log(sum_i w_i * p_i(x)).

        Parameters
        ----------
        data : Mapping[str, np.ndarray]
            Data containing inference variables and conditions.
        member_weights : Mapping[str, float] or None, default None
            Probability weights for each approximator. If None, uses uniform weights.
            Must sum to 1.
        **kwargs
            Additional arguments passed to approximator.log_prob().

        Returns
        -------
        np.ndarray
            Marginalized log probabilities with shape (batch_size,).
        """
        member_weights = self._resolve_member_weights(member_weights)
        log_probs = self.log_prob_separate(data=data, members=list(member_weights.keys()), **kwargs)

        # log p = log_sum_exp(log(w_i) + log p_i)
        stacked = np.stack(list(log_probs.values()), axis=-1)
        log_weights = np.log(list(member_weights.values()))

        # stacked: (num_datasets, num_scores), log_weights: (num_scores,)
        z = stacked + log_weights  # broadcasted to (num_datasets, num_scores)

        # stable logsumexp over last axis
        return logsumexp(z, axis=-1)

    def estimate_separate(
        self,
        conditions: Mapping[str, np.ndarray],
        members: Sequence[str] | None = None,
        split: bool = False,
        **kwargs,
    ) -> dict[str, dict[str, dict[str, np.ndarray]]]:
        """
        Compute point estimates from each approximator separately.

        Parameters
        ----------
        conditions : Mapping[str, np.ndarray]
            Conditions for estimation.
        members: Sequence[str] or None, default None
            Ensemble members to estimate with.
            If None, will estimate with all members that have an `estimate` method.
        split : bool, optional
            Whether to split output arrays, by default False.
        **kwargs
            Additional arguments passed to approximator.estimate().

        Returns
        -------
        dict[str, dict[str, dict[str, np.ndarray]]]
            Estimates keyed by approximator name, then by variable and score names.
        """
        estimates = {}
        members = list(self.approximators.keys()) if members is None else members
        for approx_name, approximator in self.approximators.items():
            if approx_name in members and hasattr(approximator, "estimate"):
                estimates[approx_name] = approximator.estimate(conditions=conditions, split=split, **kwargs)
        return estimates

    def estimate(
        self,
        conditions: Mapping[str, np.ndarray],
        members: Sequence[str] | None = None,
        split: bool = False,
        **kwargs,
    ):
        raise NotImplementedError(
            "Automatically aggregating estimates across ensemble members is not supported. "
            "Use estimate_separate() to get estimates from each approximator."
        )

    def _resolve_member_weights(self, member_weights: Mapping[str, float] | None) -> Mapping[str, float]:
        if member_weights is None:
            member_weights = {k: 1.0 for k in self.distribution_keys}
        for key, weight in member_weights.items():
            if key not in self.distribution_keys:
                raise ValueError(
                    "Member weights must be subset of self.distribution_keys. "
                    f"Unknown keys: {set(member_weights) - set(self.distribution_keys)}"
                )
            if weight < 0:
                raise ValueError(f"All member_weights must be positive. Received {key}: {weight}.")

        # Normalize weights to 1
        sum = np.sum(list(member_weights.values()))
        member_weights = {k: v / sum for k, v in member_weights.items()}
        return member_weights

    def _batch_size_from_data(self, data: Mapping[str, any]) -> int:
        """
        Fetches the current batch size from an input dictionary. Can only be used during training when
        inference variables as present.
        """
        return keras.ops.shape(data["inference_variables"])[0]

    def get_config(self):
        base_config = super().get_config()
        config = {"approximators": self.approximators}
        return base_config | serialize(config)

    @classmethod
    def from_config(cls, config, custom_objects=None):
        return cls(**deserialize(config, custom_objects=custom_objects))

    def build_from_config(self, config):
        # the approximators are already built
        pass
