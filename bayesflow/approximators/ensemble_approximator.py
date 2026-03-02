from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np
from scipy.special import logsumexp

import keras

from bayesflow.adapters import Adapter
from bayesflow.simulators import Simulator
from bayesflow.types import Tensor
from bayesflow.utils import logging, filter_kwargs
from bayesflow.utils.serialization import serializable, serialize
from bayesflow.datasets import EnsembleDataset

from .approximator import Approximator


@serializable("bayesflow.approximators")
class EnsembleApproximator(Approximator):
    def __init__(self, approximators: dict[str, Approximator], **kwargs):
        super().__init__(**kwargs)
        self._warn_if_shared_approximator_components(approximators)
        self.approximators = approximators

        self.members = tuple(self.approximators.keys())

        self.distribution_members = tuple(
            k for k, a in self.approximators.items() if getattr(a, "has_distribution", False)
        )
        self.estimate_members = tuple(k for k, a in self.approximators.items() if hasattr(a, "estimate"))

        self.has_distribution = bool(self.distribution_members)

    @classmethod
    def _warn_if_shared_approximator_components(cls, approximators):
        """Warn if approximators share component instances (not safely serializable yet)."""
        tracked = ("inference_network", "summary_network")
        seen = {name: {} for name in tracked}

        for member_name, approximator in approximators.items():
            for attr in tracked:
                if not hasattr(approximator, attr):
                    continue

                obj = getattr(approximator, attr)
                if obj is None:
                    continue

                obj_id = id(obj)
                seen[attr].setdefault(obj_id, []).append(member_name)

        # Emit one warning per shared object instance.
        for attr, by_id in seen.items():
            for members in by_id.values():
                if len(members) > 1:
                    logging.warning(
                        "EnsembleApproximator contains shared component '{attr}' across members {members}. "
                        "Deserialization of weights of shared components is not supported yet and may fail. "
                        "Use separate component instances (e.g., clone networks) to be able to serialize "
                        "the whole EnsembleApproximator object or serialize the approximators in the ensemble "
                        "separately.",
                        attr=attr,
                        members=members,
                    )

    @classmethod
    def _warn_ignored_member_weights(cls, member_weights: Mapping[str, float] | None, merge_members: bool):
        if member_weights is not None and not merge_members:
            logging.warning(
                "`member_weights` is ignored when `merge_members=False`. "
                "Set `merge_members=True` to use a weighted mixture."
            )

    @property
    def adapter(self) -> Adapter:
        # Defer to any adapter of the approximators,
        # assuming all are the same, which is not enforced at the moment.
        # self.adapter will only be used when super().fit calls build_dataset(..., adapter=self.adapter).
        return next(iter(self.approximators.values())).adapter

    def build_dataset(
        self,
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
    ) -> EnsembleDataset:
        base_dataset = super().build_dataset(
            batch_size=batch_size,
            num_batches=num_batches,
            adapter=adapter,
            memory_budget=memory_budget,
            simulator=simulator,
            workers=workers,
            use_multiprocessing=use_multiprocessing,
            max_queue_size=max_queue_size,
            **kwargs,
        )

        return EnsembleDataset(
            base_dataset=base_dataset,
            member_names=self.members,
            **filter_kwargs(kwargs, keras.utils.PyDataset.__init__),
        )

    def build(self, data_shapes: dict) -> None:
        for approx_name, approximator in self.approximators.items():
            _data_shape = {}
            for var_name, variable in data_shapes.items():  # variable type
                # If data_shapes has a nested ensemble level, select shapes with approx_name.
                # Note, summary_variables might be dict, if a FusionNetwork is used.
                # Thus, we further check whether the approx_name is in the keys.
                if isinstance(variable, dict) and approx_name in variable.keys():
                    _data_shape[var_name] = variable[approx_name]
                else:
                    _data_shape[var_name] = variable

            approximator.build(_data_shape)

    def fit(self, *args, **kwargs) -> keras.callbacks.History:
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
        **kwargs
            Additional keyword arguments passed to `keras.Model.fit()` and to the dataset constructor
            if `dataset` is not provided.

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
        inference_variables: dict[str, Tensor] | Tensor,
        inference_conditions: dict[str, Tensor] | Tensor | None = None,
        summary_variables: dict[str, Tensor] | Tensor | None = None,
        sample_weight: dict[str, Tensor] | Tensor | None = None,
        stage: str = "training",
    ) -> dict[str, dict[str, Tensor]]:
        metrics = {}

        def select(value, name):
            if value is None:
                return None
            return value[name] if stage == "training" else value

        for name, approximator in self.approximators.items():
            metrics[name] = approximator.compute_metrics(
                inference_variables=select(inference_variables, name),
                inference_conditions=select(inference_conditions, name),
                summary_variables=select(summary_variables, name),
                sample_weight=select(sample_weight, name),
                stage=stage,
            )

        metrics = {
            f"{approx_name}/{metric_key}": value
            for approx_name, approx_metrics in metrics.items()
            for metric_key, value in approx_metrics.items()
        }

        losses = [v for k, v in metrics.items() if "loss" in k]
        metrics["loss"] = keras.ops.sum(losses)

        return metrics

    def sample(
        self,
        *,
        num_samples: int,
        conditions: Mapping[str, np.ndarray],
        split: bool = False,
        member_weights: Mapping[str, float] | None = None,
        merge_members: bool = True,
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
            Must be nonnegative, will be normalized to sum to 1.
        merge_members : bool, optional
            Whether to merge samples from all approximators into a single (weighted) marginal sample.
        **kwargs
            Additional arguments passed to approximator.sample().

        Returns
        -------
        dict[str, np.ndarray]
            Samples with shape (batch_size, num_samples, ...) for each variable.
        """
        self._warn_ignored_member_weights(member_weights, merge_members)

        if not merge_members:
            return self._map_members(
                None,
                capability="distribution",
                fn=lambda name, a: a.sample(num_samples=num_samples, conditions=conditions, split=split, **kwargs),
            )

        weights = self._resolve_member_weights(member_weights)
        names = tuple(weights.keys())
        probs = np.fromiter(weights.values(), dtype=float, count=len(weights))

        counts = np.random.multinomial(num_samples, probs)
        alloc = {name: int(count) for name, count in zip(names, counts) if count > 0}

        per_member = self._map_members(
            list(alloc.keys()),
            capability="distribution",
            fn=lambda name, a: a.sample(num_samples=alloc[name], conditions=conditions, split=split, **kwargs),
        )

        merged = keras.tree.map_structure(lambda *xs: np.concatenate(xs, axis=1), *list(per_member.values()))
        idx = np.random.permutation(num_samples)

        return keras.tree.map_structure(lambda a: np.take(a, idx, axis=1), merged)

    def log_prob(
        self,
        data: Mapping[str, np.ndarray],
        member_weights: Mapping[str, float] | None = None,
        merge_members: bool = True,
        **kwargs,
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
            Must be nonnegative, will be normalized to sum to 1.
        merge_members : bool, optional
            Whether to merge log probabilities from all approximators into a single marginal log probability.
        **kwargs
            Additional arguments passed to approximator.log_prob().

        Returns
        -------
        np.ndarray
            Marginalized log probabilities with shape (batch_size,).
        """

        self._warn_ignored_member_weights(member_weights, merge_members)

        if not merge_members:
            return self._map_members(
                None,
                capability="distribution",
                fn=lambda name, a: a.log_prob(data=data, **kwargs),
            )

        weights = self._resolve_member_weights(member_weights)
        members = list(weights.keys())

        log_probs = self._map_members(
            members,
            capability="distribution",
            fn=lambda name, a: a.log_prob(data=data, **kwargs),
        )

        stacked = np.stack([log_probs[m] for m in members], axis=-1)
        log_w = np.log(np.fromiter((weights[m] for m in members), dtype=float, count=len(members)))
        return logsumexp(stacked + log_w, axis=-1)

    def estimate(
        self,
        conditions: Mapping[str, np.ndarray],
        *,
        members: Sequence[str] | None = None,
        split: bool = False,
        groupby: str = "member",
        **kwargs,
    ) -> dict:
        """
        Compute point estimates and distribution parameters from each approximator separately.

        Parameters
        ----------
        conditions : Mapping[str, np.ndarray]
            Conditions for estimation.
        members : Sequence[str] or None, default None
            Ensemble members to estimate with.
            If None, will estimate with all members that have an `estimate` method.
        split : bool, optional
            Whether to split output arrays, by default False.
        groupby : {"member", "variable"}, default "member"
            Controls the top-level nesting of the returned dictionary.

            - "member": return estimates as ``member -> variable -> score (-> head) -> array``.
            - "variable": return estimates as ``variable -> score (-> head) -> member -> array``.
              See also :py:meth:`~bayesflow.PointApproximator.estimate`.
        **kwargs
            Additional arguments passed to approximator.estimate().

        Returns
        -------
        dict[str, dict[str, dict[str, np.ndarray]]]
            Estimates keyed by approximator name, then by variable and score names.
        """
        estimates = self._map_members(
            members,
            capability="estimate",
            fn=lambda name, a: a.estimate(conditions=conditions, split=split, **kwargs),
        )

        if groupby == "member":
            return estimates
        elif groupby == "variable":
            out = {}
            for member_key, member_est in estimates.items():
                for var_key, var_est in member_est.items():
                    out.setdefault(var_key, {})

                    for score_key, score_val in var_est.items():
                        # score has heads -> dict[head] = array
                        if isinstance(score_val, dict):
                            node = out[var_key].setdefault(score_key, {})
                            if not isinstance(node, dict):
                                raise ValueError(
                                    f"Inconsistent estimate structure for variable={var_key!r}, score={score_key!r}: "
                                    "some members return a dict of heads, others return an array."
                                )
                            for head_key, arr in score_val.items():
                                node.setdefault(head_key, {})
                                node[head_key][member_key] = arr

                        # score is already an array (no head level / squeezed)
                        else:
                            node = out[var_key].setdefault(score_key, {})
                            if isinstance(node, dict) and node and any(isinstance(v, dict) for v in node.values()):
                                raise ValueError(
                                    f"Inconsistent estimate structure for variable={var_key!r}, score={score_key!r}: "
                                    "some members return an array, others return a dict of heads."
                                )
                            # keep head level absent; attach member at score level
                            if not isinstance(node, dict):
                                # should not happen, but keep it safe
                                out[var_key][score_key] = {}
                                node = out[var_key][score_key]
                            node[member_key] = score_val

            return out
        else:
            raise NotImplementedError(
                f"`groupby={groupby!r}` is not supported for EnsembleApproximator. Use 'member' or 'variable'."
            )

    def _map_members(
        self,
        members: Sequence[str] | None,
        *,
        capability: str,
        fn: Callable,
    ) -> dict[str, Any]:
        resolved = self._resolve_members(members, capability=capability)
        return {name: fn(name, self.approximators[name]) for name in resolved}

    def _resolve_members(self, members: Sequence[str] | None, *, capability: str) -> tuple[str, ...]:
        if capability == "any":
            base = self.members
        elif capability == "distribution":
            base = self.distribution_members
        elif capability == "estimate":
            base = self.estimate_members
        else:
            raise ValueError(f"Unknown capability {capability!r}")

        if members is None:
            return base

        base_set = set(base)
        members_t = tuple(members)
        unknown = [m for m in members_t if m not in base_set]
        if unknown:
            raise ValueError(f"Unknown/unsupported members for capability={capability!r}: {unknown}")
        return members_t

    def _resolve_member_weights(self, member_weights: Mapping[str, float] | None) -> Mapping[str, float]:
        if member_weights is None:
            member_weights = {k: 1.0 for k in self.distribution_members}
        for key, weight in member_weights.items():
            if key not in self.distribution_members:
                raise ValueError(
                    "Member weights must be subset of self.distribution_members. "
                    f"Unknown keys: {set(member_weights) - set(self.distribution_members)}"
                )
            if weight < 0:
                raise ValueError(f"All member_weights must be positive. Received {key}: {weight}.")

        # Normalize weights to 1
        summed = np.sum(list(member_weights.values()))
        member_weights = {k: v / summed for k, v in member_weights.items()}
        return member_weights

    def _batch_size_from_data(self, data: Mapping[str, Mapping[str, Any]]) -> int:
        """
        Fetches the current batch size from an input dictionary. Can only be used during training when
        inference variables as present.
        """
        if isinstance(data["inference_variables"], dict):
            return keras.ops.shape(data["inference_variables"][list(self.approximators.keys())[0]])[0]
        return keras.ops.shape(data["inference_variables"])[0]

    def get_config(self):
        base_config = super().get_config()
        config = {"approximators": self.approximators}
        return base_config | serialize(config)

    def build_from_config(self, config):
        # the approximators are already built
        pass
