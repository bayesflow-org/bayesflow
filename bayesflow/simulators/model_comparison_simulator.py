from collections.abc import Callable, Sequence
import numpy as np

from bayesflow.types import Shape
from bayesflow.utils import tree_concatenate
from bayesflow.utils.decorators import allow_batch_size

from bayesflow.utils import numpy_utils as npu
from bayesflow.utils import logging

from types import FunctionType
from typing import Literal

from .simulator import Simulator
from .lambda_simulator import LambdaSimulator


class ModelComparisonSimulator(Simulator):
    """Wraps a sequence of simulators for use with a model comparison approximator."""

    def __init__(
        self,
        simulators: Sequence[Simulator],
        p: Sequence[float] = None,
        logits: Sequence[float] = None,
        use_mixed_batches: bool = True,
        key_conflicts: Literal["drop", "fill", "error"] = "drop",
        fill_value: float = np.nan,
        shared_simulator: Simulator | Callable[[Sequence[int]], dict[str, any]] = None,
    ):
        """
        Initialize a multimodel simulator that can generate data for mixture / model comparison problems.

        Parameters
        ----------
        simulators : Sequence[Simulator]
            A sequence of simulator instances, each representing a different model.
        p : Sequence[float], optional
            A sequence of probabilities associated with each simulator. Must sum to 1.
            Mutually exclusive with `logits`.
        logits : Sequence[float], optional
            A sequence of logits corresponding to model probabilities. Mutually exclusive with `p`.
            If neither `p` nor `logits` is provided, defaults to uniform logits.
        use_mixed_batches : bool, optional
            Whether to draw samples in a batch from different models.

            - If True (default), each sample in a batch may come from a different model.
            - If False, the entire batch is drawn from a single model, selected according to model probabilities.
        key_conflicts : str, optional
            Policy for handling keys that are missing in the output of some models, when using mixed batches.

            - "drop" (default): Drop conflicting keys from the batch output.
            - "fill": Fill missing keys with the specified value.
            - "error": An error is raised when key conflicts are detected.
        fill_value : float, optional
            If `key_conflicts=="fill"`, the missing keys will be filled with the value of this argument.
        shared_simulator : Simulator or Callable, optional
            A shared simulator whose outputs are passed to all model simulators. If a function is
            provided, it is wrapped in a :py:class:`~bayesflow.simulators.LambdaSimulator` with batching enabled.
        """
        self.simulators = simulators

        if isinstance(shared_simulator, FunctionType):
            shared_simulator = LambdaSimulator(shared_simulator, is_batched=True)
        self.shared_simulator = shared_simulator

        match logits, p:
            case (None, None):
                logits = [0.0] * len(simulators)
            case (None, logits):
                logits = logits
            case (p, None):
                p = np.array(p)
                if not np.isclose(np.sum(p), 1.0):
                    raise ValueError("Probabilities must sum to 1.")
                logits = np.log(p) - np.log(1 - p)
            case _:
                raise ValueError("Received conflicting arguments. At most one of `p` or `logits` must be provided.")

        if len(logits) != len(simulators):
            raise ValueError(f"Length of logits ({len(logits)}) must match number of simulators ({len(simulators)}).")

        self.logits = logits
        self.use_mixed_batches = use_mixed_batches
        self.key_conflicts = key_conflicts
        self.fill_value = fill_value
        self._key_conflicts_warning = True

    @allow_batch_size
    def sample(self, batch_shape: Shape, **kwargs) -> dict[str, np.ndarray]:
        """
        Sample from the model comparison simulator.

        Parameters
        ----------
        batch_shape : Shape
            The shape of the batch to sample. Typically, a tuple indicating the number of samples,
            but the user can also supply an int.
        **kwargs
            Additional keyword arguments passed to each simulator. These may include outputs from
            the shared simulator.

        Returns
        -------
        data : dict of str to np.ndarray
            A dictionary containing the sampled outputs. Includes:
              - outputs from the selected simulator(s)
              - optionally, outputs from the shared simulator
              - "model_indices": a one-hot encoded array indicating the model origin of each sample
        """
        data = {}
        if self.shared_simulator:
            data |= self.shared_simulator.sample(batch_shape, **kwargs)

        softmax_logits = npu.softmax(self.logits)
        num_models = len(self.simulators)

        # generate data randomly from each model (slower)
        if self.use_mixed_batches:
            model_counts = np.random.multinomial(n=batch_shape[0], pvals=softmax_logits)

            sims = [
                simulator.sample(n, **(kwargs | data)) for simulator, n in zip(self.simulators, model_counts) if n > 0
            ]
            sims = self._handle_key_conflicts(sims, model_counts)
            sims = tree_concatenate(sims, numpy=True)
            data |= sims

            model_indices = np.repeat(np.eye(num_models, dtype="int32"), model_counts, axis=0)

        # draw one model index for the whole batch (faster)
        else:
            model_index = np.random.choice(num_models, p=softmax_logits)

            data = self.simulators[model_index].sample(batch_shape, **(kwargs | data))
            model_indices = npu.one_hot(np.full(batch_shape, model_index, dtype="int32"), num_models)

        return data | {"model_indices": model_indices}

    def _handle_key_conflicts(self, sims, batch_sizes):
        batch_sizes = [b for b in batch_sizes if b > 0]

        keys, all_keys, common_keys, missing_keys = self._determine_key_conflicts(sims=sims)

        # all sims have the same keys
        if all_keys == common_keys:
            return sims

        if self.key_conflicts == "drop":
            sims = [{k: v for k, v in sim.items() if k in common_keys} for sim in sims]
            return sims
        elif self.key_conflicts == "fill":
            combined_sims = {}
            for sim in sims:
                combined_sims = combined_sims | sim
            for i, sim in enumerate(sims):
                for missing_key in missing_keys[i]:
                    shape = combined_sims[missing_key].shape
                    shape = list(shape)
                    shape[0] = batch_sizes[i]
                    sim[missing_key] = np.full(shape=shape, fill_value=self.fill_value)
            return sims
        elif self.key_conflicts == "error":
            raise ValueError(
                "Different simulators provide outputs with different keys, cannot combine them into one batch."
            )

    def _determine_key_conflicts(self, sims):
        keys = [set(sim.keys()) for sim in sims]
        all_keys = set.union(*keys)
        common_keys = set.intersection(*keys)
        missing_keys = [all_keys - k for k in keys]

        if all_keys == common_keys:
            return keys, all_keys, common_keys, missing_keys

        if self._key_conflicts_warning:
            # issue warning only once
            self._key_conflicts_warning = False

            if self.key_conflicts == "drop":
                logging.info(
                    f"Incompatible simulator output. \
The following keys will be dropped: {', '.join(sorted(all_keys - common_keys))}."
                )
            elif self.key_conflicts == "fill":
                logging.info(
                    f"Incompatible simulator output. \
Attempting to replace keys: {', '.join(sorted(all_keys - common_keys))}, where missing, \
with value {self.fill_value}."
                )

        return keys, all_keys, common_keys, missing_keys
