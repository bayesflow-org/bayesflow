from collections.abc import Mapping, Sequence

import numpy as np
from scipy.special import logsumexp
import keras

from bayesflow.networks.point_inference_network import PointInferenceNetwork
from bayesflow.types import Tensor
from bayesflow.utils import filter_kwargs, split_arrays, squeeze_inner_estimates_dict, logging, concatenate_valid
from bayesflow.utils.serialization import serializable

from .continuous_approximator import ContinuousApproximator


@serializable("bayesflow.approximators")
class PointApproximator(ContinuousApproximator):
    """
    A workflow for fast amortized point estimation of a conditional distribution.

    The distribution is approximated by point estimators, parameterized by a feed-forward
    :py:class:`~bayesflow.networks.PointInferenceNetwork`. Conditions can be compressed by an optional summary network
    (inheriting from :py:class:`~bayesflow.networks.SummaryNetwork`) or used directly as input to the inference network.
    """

    def build(self, data_shapes: dict[str, tuple[int] | dict[str, dict]]) -> None:
        super().build(data_shapes)

        assert isinstance(self.inference_network, PointInferenceNetwork)

        # Infer which scoring rules induce distributions
        dist_keys = []
        for score_key, score in self.inference_network.scores.items():
            has_sample = callable(getattr(score, "sample", None))
            has_log_prob = callable(getattr(score, "log_prob", None))
            if has_sample and has_log_prob:
                dist_keys.append(score_key)
        self.distribution_keys = dist_keys

        # Update attribute to mark whether it has at least one score that represents a distribution
        if len(self.distribution_keys) == 0:
            self.has_distribution = False

    def estimate_separate(
        self,
        conditions: Mapping[str, np.ndarray],
        split: bool = False,
        **kwargs,
    ) -> dict[str, dict[str, dict[str, np.ndarray]]]:
        """
        Estimates point summaries of inference variables for each parametric scoring rule separately.

        This method processes input conditions, computes estimates, applies necessary adapter transformations,
        and optionally splits the resulting arrays along the last axis. Estimates are grouped by scoring rule,
        then estimation head, then inference variable.

        Parameters
        ----------
        conditions : Mapping[str, np.ndarray]
            A dictionary mapping variable names to arrays representing the conditions
            for the estimation process.
        split : bool, optional
            If True, the estimated arrays are split along the last axis, by default False.
        **kwargs
            Additional keyword arguments passed to underlying processing functions.

        Returns
        -------
        dict[str, dict[str, dict[str, np.ndarray]]]
            Nested dictionary where keys follow ``score -> estimator head -> inference variable`` ordering.
            Each estimator output (i.e., dictionary value that is not itself a dictionary) is an array of shape
            ``(num_datasets, point_estimate_size, ...)``.
        """
        # Adapt, optionally standardize and convert conditions to tensor.
        conditions = self._prepare_data(conditions, **kwargs)
        # Remove any superfluous keys, just retain actual conditions.  # TODO: is this necessary?
        conditions = {k: v for k, v in conditions.items() if k in self.CONDITION_KEYS}

        estimates = self._estimate(**conditions, **kwargs)

        if "inference_variables" in self.standardize:
            for score_key, score in self.inference_network.scores.items():
                for head_key in estimates[score_key].keys():
                    transformation_type = score.TRANSFORMATION_TYPE.get(head_key, "location_scale")
                    estimates[score_key][head_key] = self.standardize_layers["inference_variables"](
                        estimates[score_key][head_key], forward=False, transformation_type=transformation_type
                    )

        estimates = self._apply_inverse_adapter_to_estimates(estimates, **kwargs)

        # Optionally split the arrays along the last axis.
        if split:
            estimates = split_arrays(estimates, axis=-1)

        return estimates

    def estimate(
        self,
        conditions: Mapping[str, np.ndarray],
        split: bool = False,
        **kwargs,
    ) -> dict[str, dict[str, np.ndarray | dict[str, np.ndarray]]]:
        """
        Estimates point summaries of inference variables based on specified conditions.

        This method delegates to :meth:`estimate_separate`, then reorders and squeezes the
        resulting dictionary so that inference variable names become the top-level keys.

        Parameters
        ----------
        conditions : Mapping[str, np.ndarray]
            A dictionary mapping variable names to arrays representing the conditions
            for the estimation process.
        split : bool, optional
            If True, the estimated arrays are split along the last axis, by default False.
        **kwargs
            Additional keyword arguments passed to underlying processing functions.

        Returns
        -------
        dict[str, dict[str, np.ndarray or dict[str, np.ndarray]]]
            Nested dictionary ordered by ``inference variable -> scoring rule``.
            Entries have shape (num_datasets, ...).
        """
        estimates = self.estimate_separate(conditions=conditions, split=split, **kwargs)

        # Reorder the nested dictionary so that original variable names are at the top.
        estimates = self._reorder_estimates(estimates)
        # Remove unnecessary nesting.
        estimates = self._squeeze_estimates(estimates)

        return estimates

    def sample_separate(
        self,
        *,
        num_samples: int,
        conditions: Mapping[str, np.ndarray],
        split: bool = False,
        **kwargs,
    ) -> dict[str, dict[str, np.ndarray]]:
        """
        Draws samples from parametric distributions based on parameter estimates for given input conditions.

        These samples generally do not correspond to samples from the fully Bayesian posterior, since
        they assume some parametric form (e.g., multivariate normal when using the MultivariateNormalScore).

        Parameters
        ----------
        num_samples : int
            The number of samples to generate.
        conditions : Mapping[str, np.ndarray]
            A dictionary mapping variable names to arrays representing the conditions
            for the sampling process.
        split : bool, optional
            If True, the sampled arrays are split along the last axis, by default False.
            Currently not supported for :py:class:`PointApproximator` .
        **kwargs
            Additional keyword arguments passed to underlying processing functions.

        Returns
        -------
        samples : dict[str, np.ndarray or dict[str, np.ndarray]]
            Samples for all inference variables and all parametric scoring rules in a nested dictionary.

            1. Each first-level key is the name of an inference variable.
            2. Each second-level key is the name of a parametric score.

            Each output (i.e., dictionary value that is not itself a dictionary) is an array
            of shape (num_datasets, num_samples, ...).
        """
        self._check_has_distribution()

        # Adapt, optionally standardize and convert conditions to tensor.
        conditions = self._prepare_data(conditions, **kwargs)
        # Remove any superfluous keys, just retain actual conditions.  # TODO: is this necessary?
        conditions = {k: v for k, v in conditions.items() if k in self.CONDITION_KEYS}

        # Sample and undo optional standardization
        samples = self._sample(num_samples, **conditions, **kwargs)

        if "inference_variables" in self.standardize:
            for score_key in samples.keys():
                samples[score_key] = self.standardize_layers["inference_variables"](samples[score_key], forward=False)

        samples = self._apply_inverse_adapter_to_samples(samples, **kwargs)

        if split:
            raise NotImplementedError("split=True is currently not supported for `PointApproximator`.")

        return samples

    def sample(
        self,
        *,
        num_samples: int,
        conditions: Mapping[str, np.ndarray],
        split: bool = False,
        score_weights: Sequence[float] | Mapping[str, float] | None = None,
        **kwargs,
    ) -> dict[str, np.ndarray]:
        """
        Draws samples from the mixture induced by all parametric scoring rules.

        Samples are allocated to scoring rules via multinomial sampling using score_weights,
        then drawn efficiently by sampling only max(num_samples_per_score) per score, cropping
        to the allocated counts, concatenating along the sample axis, and finally shuffling.

        Parameters
        ----------
        num_samples : int
            Total number of samples to draw.
        conditions : Mapping[str, np.ndarray]
            Conditions for sampling.
        split : bool, optional
            Whether to split the output arrays along the last axis. Delegated to :meth:`sample_separate`.
        score_weights : Sequence[float] or Mapping[str, float], optional
            Probability weights for each scoring rule. If ``None``, uniform weights are assumed. The weights must
            sum to 1 and are ordered according to the scoring rule keys returned by :meth:`sample_separate` unless
            provided as a mapping.
        **kwargs
            Additional keyword arguments forwarded to :meth:`sample_separate`.

        Returns
        -------
        dict[str, np.ndarray]
            Samples aggregated across all scoring rules, keyed by inference variable names.
            Entries have shape (num_datasets, num_samples, ...).
        """
        self._check_has_distribution()

        score_weights = self._resolve_score_weights(score_weights)

        # Allocate samples per score and draw only as many as needed (max over scores).
        num_samples_per_score = np.random.multinomial(num_samples, score_weights)
        max_k = int(np.max(num_samples_per_score))

        samples_by_score = self.sample_separate(num_samples=max_k, conditions=conditions, split=split, **kwargs)

        # Crop each score's samples down to its allocated k
        cropped_list = []
        for score_key, k in zip(self.distribution_keys, num_samples_per_score):
            if k == 0:
                continue
            cropped = keras.tree.map_structure(lambda arr: arr[:, :k], samples_by_score[score_key])
            cropped_list.append(cropped)

        # Concatenate across scores along the sample axis.
        concatenated = keras.tree.map_structure(
            lambda *arrays: np.concatenate(arrays, axis=1),
            *cropped_list,
        )

        # Shuffle along the sample axis (1) to form the mixture samples.
        shuffle_idx = np.random.permutation(num_samples)
        shuffled = keras.tree.map_structure(lambda arr: np.take(arr, shuffle_idx, axis=1), concatenated)
        return shuffled

    def log_prob_separate(self, data: Mapping[str, np.ndarray], **kwargs) -> dict[str, np.ndarray]:
        """
        Computes the log-probability of given data under the parametric distribution(s) for given input conditions.

        Parameters
        ----------
        data : dict[str, np.ndarray]
            A dictionary mapping variable names to arrays representing the inference conditions and variables.
        **kwargs
            Additional keyword arguments passed to underlying processing functions.

        Returns
        -------
        log_prob : dict[str, np.ndarray]
            Log-probabilities of the distribution
            `p(inference_variables | inference_conditions, h(summary_conditions))` for all parametric scoring rules.
            Each has shape (num_datasets,).
        """
        self._check_has_distribution()

        # Adapt, optionally standardize and convert to tensor. Keep track of log_det_jac
        data, log_det_jac = self._prepare_data(data, log_det_jac=True, **kwargs)

        # Pass data to networks and convert back to numpy array
        log_prob = self._log_prob(**data, **kwargs)
        log_prob = keras.tree.map_structure(keras.ops.convert_to_numpy, log_prob)

        # Change of variables formula, respecting log_prob to be a dictionary
        if log_det_jac is not None:
            log_prob = keras.tree.map_structure(lambda x: x + log_det_jac, log_prob)

        return log_prob

    def log_prob(
        self,
        data: Mapping[str, np.ndarray],
        score_weights: Sequence[float] | Mapping[str, float] | None = None,
        **kwargs,
    ) -> np.ndarray:
        """
        Computes the marginalized log-probability across all parametric scoring rules.

        Parameters
        ----------
        data : Mapping[str, np.ndarray]
            Dictionary containing inference variables and conditions.
        score_weights : Sequence[float] or Mapping[str, float], optional
            Probability weights for each scoring rule. If ``None``, uniform weights are assumed. The weights must sum
            to 1 and are ordered by the scoring rule keys produced by :meth:`log_prob_separate` unless provided as a
            mapping.
        **kwargs
            Additional keyword arguments forwarded to :meth:`log_prob_separate`.

        Returns
        -------
        np.ndarray, shape (num_datasets,)
            Marginalized log-probabilities with the same leading dimensions as the inputs.
        """
        log_probs = self.log_prob_separate(data=data, **kwargs)

        score_weights = self._resolve_score_weights(score_weights)

        stacked = np.stack([np.asarray(log_probs[score_key]) for score_key in self.distribution_keys], axis=-1)
        log_weights = np.log(score_weights)

        # stacked: (num_datasets, num_scores), log_weights: (num_scores,)
        z = stacked + log_weights  # broadcasted to (num_datasets, num_scores)

        # stable logsumexp over last axis
        return logsumexp(z)

    def _check_has_distribution(self):
        if not self.has_distribution:
            raise ValueError("No parametric distribution scores available for sample/log_prob.")

    def _resolve_score_weights(
        self,
        score_weights: Sequence[float] | Mapping[str, float] | None,
    ) -> np.ndarray:
        if score_weights is None:
            return np.ones(len(self.distribution_keys), dtype=np.float64) / len(self.distribution_keys)

        if isinstance(score_weights, Mapping):
            missing = set(self.distribution_keys) - set(score_weights.keys())
            if missing:
                raise ValueError(f"score_weights is missing entries for scoring rules: {sorted(missing)}")
            extra = set(score_weights.keys()) - set(self.distribution_keys)
            if extra:
                raise ValueError(f"score_weights contains unknown scoring rules: {sorted(extra)}")
            weights = np.asarray([score_weights[key] for key in self.distribution_keys], dtype=np.float64)
        else:
            weights = np.asarray(score_weights, dtype=np.float64)
            if weights.ndim != 1 or weights.shape[0] != len(self.distribution_keys):
                raise ValueError(
                    f"score_weights must be a 1D array with length {len(self.distribution_keys)}; "
                    f"received shape {weights.shape}."
                )
        total = np.sum(weights)
        if not np.isclose(total, 1.0, atol=1e-8):
            raise ValueError(f"score_weights must sum to 1: {weights} -> {total}.")

        return weights

    def _apply_inverse_adapter_to_estimates(
        self, estimates: Mapping[str, Mapping[str, Tensor]], **kwargs
    ) -> dict[str, dict[str, dict[str, np.ndarray]]]:
        """Applies the inverse adapter on each inner element of the _estimate output dictionary."""
        estimates = keras.tree.map_structure(keras.ops.convert_to_numpy, estimates)
        processed = {}
        for score_key, score_val in estimates.items():
            processed[score_key] = {}
            for head_key, estimate in score_val.items():
                if head_key in self.inference_network.scores[score_key].NOT_TRANSFORMING_LIKE_VECTOR_WARNING:
                    logging.warning(
                        f"Estimate '{score_key}.{head_key}' is marked to not transform like a vector. "
                        f"It was treated like a vector by the adapter. Handle '{head_key}' estimates with care."
                    )

                adapted = self.adapter(
                    {"inference_variables": estimate},
                    inverse=True,
                    strict=False,
                    **kwargs,
                )
                processed[score_key][head_key] = adapted
        return processed

    def _apply_inverse_adapter_to_samples(
        self, samples: Mapping[str, Tensor], **kwargs
    ) -> dict[str, dict[str, np.ndarray]]:
        """Applies the inverse adapter to a dictionary of samples."""
        samples = keras.tree.map_structure(keras.ops.convert_to_numpy, samples)
        processed = {}
        for score_key, score_value in samples.items():
            processed[score_key] = self.adapter(
                {"inference_variables": score_value},
                inverse=True,
                strict=False,
                **kwargs,
            )
        return processed

    @staticmethod
    def _reorder_estimates(
        estimates: Mapping[str, Mapping[str, Mapping[str, np.ndarray]]],
    ) -> dict[str, dict[str, dict[str, np.ndarray]]]:
        """Reorders the nested dictionary so that the inference variable names become the top-level keys."""
        # Grab the variable names from one sample inner dictionary.
        sample_inner = next(iter(next(iter(estimates.values())).values()))
        variable_names = sample_inner.keys()
        reordered = {}
        for variable in variable_names:
            reordered[variable] = {}
            for score_key, inner_dict in estimates.items():
                reordered[variable][score_key] = {inner_key: value[variable] for inner_key, value in inner_dict.items()}
        return reordered

    @staticmethod
    def _squeeze_estimates(
        estimates: Mapping[str, Mapping[str, Mapping[str, np.ndarray]]],
    ) -> dict[str, dict[str, np.ndarray]]:
        """Squeezes each inner estimate dictionary to remove unnecessary nesting."""
        squeezed = {}
        for variable, variable_estimates in estimates.items():
            squeezed[variable] = {
                score_key: squeeze_inner_estimates_dict(inner_estimate)
                for score_key, inner_estimate in variable_estimates.items()
            }
        return squeezed

    def _estimate(
        self,
        inference_conditions: Tensor = None,
        summary_variables: Tensor = None,
        **kwargs,
    ) -> dict[str, dict[str, Tensor]]:
        if (self.summary_network is None) != (summary_variables is None):
            raise ValueError("Summary variables and summary network must be used together.")

        if self.summary_network is not None:
            summary_outputs = self.summary_network(
                summary_variables, **filter_kwargs(kwargs, self.summary_network.call)
            )
            inference_conditions = concatenate_valid((inference_conditions, summary_outputs), axis=-1)

        return self.inference_network(
            conditions=inference_conditions,
            **filter_kwargs(kwargs, self.inference_network.call),
        )
