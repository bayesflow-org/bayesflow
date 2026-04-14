from collections.abc import Sequence, Callable
from typing import Literal, Tuple

import time
import copy

import numpy as np
import keras

from bayesflow.adapters import Adapter
from bayesflow.approximators import CompositionalApproximator
from bayesflow.networks import InferenceNetwork, SummaryNetwork, DiffusionModel
from bayesflow.simulators import Simulator
from bayesflow.types import Tensor
from bayesflow.utils import find_inference_network, find_summary_network, logging, format_duration, filter_kwargs

from .basic_workflow import BasicWorkflow


class CompositionalWorkflow(BasicWorkflow):
    """
    This class extends the Basic Workflow to support compositional inference, allowing for the generation of
    samples conditioned on multiple datasets or compositional conditions.

    Parameters
    ----------
    simulator : Simulator, optional
        A Simulator object to generate synthetic data for inference (default is None).
    adapter : Adapter, optional
        Adapter for data processing. If not provided, a default adapter will be used (default is None), but
        you need to make sure to provide the correct names for `inference_variables` and/or `inference_conditions`
        and/or `summary_variables`.
    inference_network : InferenceNetwork or str, optional
        The inference network used for posterior approximation, specified as an instance or by
        name (default is "diffusion_model").
    summary_network : SummaryNetwork or str, optional
        The summary network used for data summarization, specified as an instance or by name (default is None).
    initial_learning_rate : float, optional
        Initial learning rate for the optimizer (default is 5e-4).
    optimizer : type, optional
        The optimizer to be used for training. If None, a default Adam optimizer will be selected (default is None).
    checkpoint_filepath : str, optional
        Directory path where model checkpoints will be saved (default is None).
    checkpoint_name : str, optional
        Name of the checkpoint file (default is "model").
    save_weights_only : bool, optional
        If True, only the model weights will be saved during checkpointing (default is False).
    save_best_only: bool, optional
        If only the best model according to the quantity monitored (loss or validation) at the end of
        each epoch will be saved instead of the last model (default is False). Use with caution,
        as some losses (e.g. flow matching) do not reliably reflect model performance, and outliers in the
        validation data can cause unwanted effects.
    inference_variables : Sequence[str] or str, optional
        Variables for inference as a sequence of strings or a single string (default is None).
        Important for automating diagnostics!
    inference_conditions : Sequence[str] or str, optional
        Variables used as direct conditions for inference (default is None).
    summary_variables : Sequence[str] or str, optional
        Variables to be summarized through the summary network before being used as conditions (default is None).
    standardize : Sequence[str] or str, optional
        Variables to standardize during preprocessing (default is "inference_variables"). These will be
        passed to the corresponding approximator constructor and can be either "all" or any subset of
        ["inference_variables", "summary_variables", "inference_conditions"].
    **kwargs : dict, optional
        Additional arguments for configuring networks, adapters, optimizers, etc.
    """

    def __init__(
        self,
        simulator: Simulator | None = None,
        adapter: Adapter | None = None,
        inference_network: InferenceNetwork | str = "diffusion_model",
        summary_network: SummaryNetwork | str | None = None,
        initial_learning_rate: float = 5e-4,
        optimizer: keras.optimizers.Optimizer | type | None = None,
        checkpoint_filepath: str | None = None,
        checkpoint_name: str = "model",
        save_weights_only: bool = False,
        save_best_only: bool = False,
        inference_variables: Sequence[str] | str | None = None,
        inference_conditions: Sequence[str] | str | None = None,
        summary_variables: Sequence[str] | str | None = None,
        standardize: Sequence[str] | str | None = "inference_variables",
        **kwargs,
    ):
        self.inference_network = find_inference_network(inference_network, **kwargs.get("inference_kwargs", {}))

        if not isinstance(self.inference_network, DiffusionModel):
            raise ValueError("Inference network currently must be a DiffusionModel for compositional inference.")

        if summary_network is not None:
            self.summary_network = find_summary_network(summary_network, **kwargs.get("summary_kwargs", {}))
        else:
            self.summary_network = None

        self.simulator = simulator

        adapter = adapter or BasicWorkflow.default_adapter(inference_variables, inference_conditions, summary_variables)

        self.approximator = CompositionalApproximator(
            inference_network=self.inference_network,
            summary_network=self.summary_network,
            adapter=adapter,
            standardize=standardize,
            **filter_kwargs(kwargs, keras.Model.__init__),
        )

        self._init_optimizer(initial_learning_rate, optimizer, **kwargs.get("optimizer_kwargs", {}))
        self._init_checkpointing(checkpoint_filepath, checkpoint_name, save_weights_only, save_best_only)
        self.history = None

    def compositional_sample(
        self,
        *,
        num_samples: int,
        conditions: dict[str, np.ndarray] | None = None,
        compute_prior_score: Callable[[dict[str, np.ndarray], np.ndarray], dict[str, np.ndarray]] = None,
        summaries: Tensor | np.ndarray | None = None,
        split: bool = False,
        batch_size: int | None = None,
        sample_shape: Literal["infer"] | Tuple[int] | int = "infer",
        **kwargs,
    ) -> dict[str, np.ndarray]:
        """
        Draws `num_samples` samples from the approximator given specified composition conditions.
        The `conditions` dictionary should have shape (n_datasets, n_compositional_conditions, ...).

        Parameters
        ----------
        num_samples : int
            The number of samples to generate.
        conditions : dict[str, np.ndarray], optional
            A dictionary where keys represent variable names and values are
            NumPy arrays containing the adapted simulated variables. Keys used as summary or inference
            conditions during training should be present.
            Should have shape (n_datasets, n_compositional_conditions, ...).
        compute_prior_score : Callable[[dict[str, np.ndarray], np.ndarray], dict[str, np.ndarray]], optional
            A function that computes the score of the log prior distribution.
            Otherwise, the unconditional score is used.
        summaries : Tensor | np.ndarray | None, optional
            Precomputed summary outputs to be used as conditions for sampling. If provided, these will be used instead
            of the conditions. Should have shape (n_datasets, n_compositional_conditions, ...).
        split : bool, default=False
            Whether to split the output arrays along the last axis and return one sample array per target variable.
        batch_size : int or None, optional
            If provided, the conditions are split into batches of size `batch_size`, for which samples are generated
            sequentially. Can help with memory management for large sample sizes.
        sample_shape : str or tuple of int, optional
            Trailing structural dimensions of each generated sample, excluding the batch and target (intrinsic)
            dimension. For example, use `(time,)` for time series or `(height, width)` for images.

            If set to `"infer"` (default), the structural dimensions are inferred from the `inference_conditions`.
            In that case, all non-vector dimensions except the last (channel) dimension are treated as structural
            dimensions. For example, if the final `inference_conditions` have shape `(batch_size, time, channels)`,
            then `sample_shape` is inferred as `(time,)`, and the generated samples will have shape
            `(num_conditions, num_samples, time, target_dim)`.
        **kwargs : dict
            Additional keyword arguments passed to the approximator's sampling function.

        Returns
        -------
        dict[str, np.ndarray]
            A dictionary where keys correspond to variable names and
            values are arrays containing the generated samples.
        """
        start_time = time.perf_counter()
        samples = self.approximator.compositional_sample(
            num_samples=num_samples,
            conditions=conditions,
            compute_prior_score=compute_prior_score,
            split=split,
            batch_size=batch_size,
            sample_shape=sample_shape,
            summary_outputs=summaries,
            **kwargs,
        )
        elapsed = time.perf_counter() - start_time
        logging.info(f"Sampling completed in {format_duration(elapsed)}.")
        return samples

    @classmethod
    def from_basic_workflow(
        cls,
        workflow: BasicWorkflow,
        **kwargs,
    ) -> "CompositionalWorkflow":
        """
        Build a :class:`CompositionalWorkflow` from a trained :class:`BasicWorkflow`.

        The trained ``DiffusionModel`` inference network (and, if present, the summary
        network) are transferred directly so no re-training is needed.

        Parameters
        ----------
        workflow : BasicWorkflow
            A fitted workflow whose ``approximator.inference_network`` is a
            :class:`~bayesflow.networks.DiffusionModel`.
        **kwargs
            Override any constructor argument of :class:`CompositionalWorkflow`,
            e.g. ``optimizer``, ``simulator``, ``adapter``, etc.

            The following attributes pertaining to checkpointing will not be
            transferred from the source workflow:
            - ``checkpoint_filepath``
            - ``checkpoint_name``
            - ``save_weights_only``
            - ``save_best_only``

            They can be set via kwargs if needed.

        Returns
        -------
        compositional_workflow: CompositionalWorkflow
            The newly created compositional workflow with attributes from ``workflow`` and
            ``kwargs``.
        """
        if not isinstance(workflow, BasicWorkflow):
            raise TypeError(f"Expected a BasicWorkflow instance, got {type(workflow).__name__!r}.")

        approximator = workflow.approximator

        # Clone the networks so the two workflows have independent weights
        cloned_inference_network = keras.models.clone_model(approximator.inference_network)
        cloned_inference_network.set_weights(approximator.inference_network.get_weights())

        if not isinstance(approximator.inference_network, DiffusionModel):
            raise ValueError(
                f"The inference network must be a DiffusionModel for compositional inference, "
                f"got {type(approximator.inference_network).__name__!r}."
            )

        if approximator.summary_network is not None:
            cloned_summary_network = keras.models.clone_model(approximator.summary_network)
            cloned_summary_network.set_weights(approximator.summary_network.get_weights())
        else:
            cloned_summary_network = approximator.summary_network

        # Collect all attributes from the basic workflow that can be passed to the constructor.
        init_kwargs = dict(
            simulator=workflow.simulator,
            adapter=approximator.adapter,
            inference_network=cloned_inference_network,
            summary_network=cloned_summary_network,
            initial_learning_rate=workflow.initial_learning_rate,
            optimizer=workflow.optimizer,
            standardize=approximator.standardizer.standardize,
        )

        # Override with caller-supplied kwargs and create new workflow
        compositional_workflow = cls(**(init_kwargs | kwargs))

        # Replace the fresh (unfitted) standardizer with a deep copy of the source one
        compositional_workflow.approximator.standardizer = copy.deepcopy(approximator.standardizer)

        return compositional_workflow
