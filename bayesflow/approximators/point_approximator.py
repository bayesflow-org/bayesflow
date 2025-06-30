from collections.abc import Mapping

import numpy as np

import keras

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

    def estimate(
        self,
        conditions: Mapping[str, np.ndarray],
        split: bool = False,
        **kwargs,
    ) -> dict[str, dict[str, np.ndarray | dict[str, np.ndarray]]]:
        """
        Estimates point summaries of inference variables based on specified conditions.

        This method processes input conditions, computes estimates, applies necessary adapter transformations,
        and optionally splits the resulting arrays along the last axis.

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
        estimates : dict[str, dict[str, np.ndarray or dict[str, np.ndarray]]]
            The estimates of inference variables in a nested dictionary.

            1. Each first-level key is the name of an inference variable.
            2. Each second-level key is the name of a scoring rule.
            3. (If the scoring rule comprises multiple estimators, each third-level key is the name of an estimator.)

            Each estimator output (i.e., dictionary value that is not itself a dictionary) is an array
            of shape (num_datasets, point_estimate_size, variable_block_size).
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

        # Reorder the nested dictionary so that original variable names are at the top.
        estimates = self._reorder_estimates(estimates)
        # Remove unnecessary nesting.
        estimates = self._squeeze_estimates(estimates)

        return estimates

    def sample(
        self,
        *,
        num_samples: int,
        conditions: Mapping[str, np.ndarray],
        split: bool = False,
        **kwargs,
    ) -> dict[str, dict[str, np.ndarray]]:
        """
        Draws samples from a parametric distribution based on point estimates for given input conditions.

        These samples will generally not correspond to samples from the fully Bayesian posterior, since
        they will assume some parametric form (e.g., multivariate normal when using the MultivariateNormalScore).

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
            2. (If there are multiple parametric scores, each second-level key is the name of such a score.)

            Each output (i.e., dictionary value that is not itself a dictionary) is an array
            of shape (num_datasets, num_samples, variable_block_size).
        """
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

        # Squeeze sample dictionary if there's only one key-value pair.
        samples = self._squeeze_parametric_score_major_dict(samples)

        return samples

    def log_prob(
        self,
        *,
        data: Mapping[str, np.ndarray],
        **kwargs,
    ) -> np.ndarray | dict[str, np.ndarray]:
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
        log_prob : np.ndarray or dict[str, np.ndarray]
            Log-probabilities of the distribution
            `p(inference_variables | inference_conditions, h(summary_conditions))` for all parametric scoring rules.

            If only one parametric score is available, output is an array of log-probabilities.

            Output is a dictionary if multiple parametric scores are available.
            Then, each key is the name of a score and values are corresponding log-probabilities.

            Log-probabilities have shape (num_datasets,).
        """
        # Adapt, optionally standardize and convert to tensor. Keep track of log_det_jac
        data, log_det_jac = self._prepare_data(data, log_det_jac=True, **kwargs)

        # Pass data to networks and convert back to numpy array
        log_prob = self._log_prob(**data, **kwargs)
        log_prob = keras.tree.map_structure(keras.ops.convert_to_numpy, log_prob)

        # Change of variables formula, respecting log_prob to be a dictionary
        if log_det_jac is not None:
            log_prob = keras.tree.map_structure(lambda x: x + log_det_jac, log_prob)

        log_prob = self._squeeze_parametric_score_major_dict(log_prob)

        return log_prob

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

    @staticmethod
    def _squeeze_parametric_score_major_dict(samples: Mapping[str, np.ndarray]) -> np.ndarray or dict[str, np.ndarray]:
        """Squeezes the dictionary to just the value if there is only one key-value pair."""
        if len(samples) == 1:
            # Extract and return the only item's value
            return next(iter(samples.values()))
        return samples

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
