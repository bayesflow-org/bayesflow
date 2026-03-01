from collections.abc import Mapping

import numpy as np

import keras

from bayesflow.types import Tensor
from bayesflow.utils import (
    logging,
    filter_kwargs,
    split_arrays,
    squeeze_inner_estimates_dict,
)
from bayesflow.utils.serialization import serializable

from .continuous_approximator import ContinuousApproximator


@serializable("bayesflow.approximators")
class PointApproximator(ContinuousApproximator):
    """
    A workflow for fast amortized point estimation of a conditional distribution.

    Inherits from :class:`ContinuousApproximator` and adapts the sample, log_prob, and estimate
    interfaces for the nested output structure of :class:`~bayesflow.networks.PointInferenceNetwork`.

    Parameters
    ----------
    adapter : bayesflow.adapters.Adapter
        Adapter for data processing. You can use :py:meth:`build_adapter` to create it.
    inference_network : InferenceNetwork
        The inference network used for point estimation.
    summary_network : SummaryNetwork, optional
        The summary network used for data summarization (default is None).
    standardize : str | Sequence[str] | None
        The variables to standardize before passing to the networks. Can be either
        "all" or any subset of ["inference_variables", "summary_variables", "inference_conditions"].
        (default is "inference_variables").
    **kwargs : dict, optional
        Additional arguments passed to the :py:class:`ContinuousApproximator` class.
    """

    def estimate(
        self,
        conditions: Mapping[str, np.ndarray],
        split: bool = False,
        **kwargs,
    ) -> dict[str, dict[str, np.ndarray | dict[str, np.ndarray]]]:
        """
        Estimates point summaries of inference variables based on specified conditions.

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

            Each estimator output is an array of shape (num_datasets, point_estimate_size, variable_block_size).
        """
        # Adapt, standardize, and resolve conditions
        resolved_conditions = self._prepare_conditions(conditions)[0]

        # Compute point estimates
        estimates = self.inference_network(
            conditions=resolved_conditions,
            **filter_kwargs(kwargs, self.inference_network.call),
        )

        # Unstandardize the network outputs
        if "inference_variables" in self.standardizer.standardize:
            for score_key, score in self.inference_network.scores.items():
                for head_key in estimates[score_key].keys():
                    transformation_type = score.TRANSFORMATION_TYPE.get(head_key, "location_scale")
                    estimates[score_key][head_key] = self.standardizer.maybe_standardize(
                        estimates[score_key][head_key],
                        key="inference_variables",
                        stage="inference",
                        forward=False,
                        transformation_type=transformation_type,
                    )

        # Apply inverse adapter to each estimate
        estimates = self._apply_inverse_adapter_to_estimates(estimates, **kwargs)

        if split:
            estimates = split_arrays(estimates, axis=-1)

        estimates = self._reorder_estimates(estimates)
        estimates = self._squeeze_estimates(estimates)

        return estimates

    def sample(
        self,
        *,
        num_samples: int,
        conditions: Mapping[str, np.ndarray],
        split: bool = False,
        batch_size: int | None = None,
        **kwargs,
    ) -> dict[str, np.ndarray | dict[str, np.ndarray]]:
        """
        Draws samples from a parametric distribution based on point estimates.

        Uses :meth:`ContinuousApproximator.sample` for condition resolution, sampling,
        unstandardization, and inverse adapter, then squeezes the nested score-major
        output structure.

        Parameters
        ----------
        num_samples : int
            The number of samples to generate.
        conditions : Mapping[str, np.ndarray]
            A dictionary mapping variable names to arrays representing the conditions.
        split : bool, optional
            If True, the sampled arrays are split along the last axis, by default False.
            Currently not supported for :py:class:`PointApproximator`.
        batch_size : int or None, optional
            If provided, the conditions are split into batches of size `batch_size`,
            for which samples are generated sequentially.
        **kwargs
            Additional keyword arguments passed to underlying processing functions.

        Returns
        -------
        samples : dict[str, np.ndarray or dict[str, np.ndarray]]
            Samples for all inference variables and all parametric scoring rules in a nested dictionary.
            Shape: (num_datasets, num_samples, variable_block_size).
        """
        if split:
            raise NotImplementedError("split=True is currently not supported for `PointApproximator`.")

        # Delegate to parent for condition resolution, sampling, unstandardization, and inverse adapter
        samples = super().sample(
            num_samples=num_samples,
            conditions=conditions,
            split=False,
            batch_size=batch_size,
            **kwargs,
        )

        return self._squeeze_parametric_score_major_dict(samples)

    def log_prob(self, data: Mapping[str, np.ndarray], **kwargs) -> np.ndarray | dict[str, np.ndarray]:
        """
        Computes the log-probability of given data under the parametric distribution(s).

        Parameters
        ----------
        data : dict[str, np.ndarray]
            A dictionary mapping variable names to arrays representing the data.
        **kwargs
            Additional keyword arguments passed to underlying processing functions.

        Returns
        -------
        log_prob : np.ndarray or dict[str, np.ndarray]
            Log-probabilities of the distribution for all parametric scoring rules.
            If only one parametric score is available, returns an array.
            Otherwise, returns a dictionary with score names as keys.
            Shape: (num_datasets,)
        """
        log_prob = super().log_prob(data, **kwargs)
        return self._squeeze_parametric_score_major_dict(log_prob)

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

    @staticmethod
    def _reorder_estimates(
        estimates: Mapping[str, Mapping[str, Mapping[str, np.ndarray]]],
    ) -> dict[str, dict[str, dict[str, np.ndarray]]]:
        """Reorders the nested dictionary so that the inference variable names become the top-level keys."""
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
    def _squeeze_parametric_score_major_dict(samples: Mapping[str, np.ndarray]) -> np.ndarray | dict[str, np.ndarray]:
        """Squeezes the dictionary to just the value if there is only one key-value pair."""
        if len(samples) == 1:
            return next(iter(samples.values()))
        return samples
