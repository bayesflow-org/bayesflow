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
    """A multimodel simulator useful for model comparison tasks.

    This class wraps multiple :class:`~bayesflow.simulators.Simulator` instances and
    produces batched outputs that include a one-hot ``inference_variables`` vector
    indicating which simulator generated each sample. It supports two sampling
    modes:

    * **mixed batches** (default) - each element in the batch may originate from
      a different simulator; the number of draws per model is drawn from a
      multinomial with probabilities given by ``softmax(logits)``.
    * **single-model batches** - the entire batch is drawn from a single
      simulator chosen according to the model probabilities.

    A shared simulator may optionally provide additional data that is passed to
    every model's sampling call. Key-conflict policies control how incompatible
    outputs across simulators are handled (drop, fill, or error).

    Parameters
    ----------
    simulators : Sequence[Simulator]
        A sequence of simulator instances, each representing a different model.
    p : Sequence[float], optional
        A sequence of probabilities associated with each simulator. Must sum to 1.
        Mutually exclusive with ``logits``.
    logits : Sequence[float], optional
        A sequence of logits corresponding to model probabilities. Mutually
        exclusive with ``p``. If neither ``p`` nor ``logits`` is provided,
        uniform logits are assumed.
    use_mixed_batches : bool, optional
        Whether to draw samples in a batch from different models.

        * If ``True`` (default), each sample in a batch may come from a different
          model.
        * If ``False``, the entire batch is drawn from a single model selected
          according to the model probabilities.
    key_conflicts : {"drop", "fill", "error"}, optional
        Policy for handling keys missing from some model outputs when mixing
        batches.

        * ``"drop"`` (default): drop conflicting keys from the batch output.
        * ``"fill"``: fill missing keys with ``fill_value``.
        * ``"error"``: raise an error on conflicts.
    fill_value : float, optional
        If ``key_conflicts=="fill"``, missing keys are filled with this value.
    shared_simulator : Simulator or Callable, optional
        A shared simulator providing outputs to every model. If a callable is
        passed it is wrapped in a
        :class:`~bayesflow.simulators.LambdaSimulator` with batching enabled.
    """

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
        # constructor body unchanged
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
              - "inference_variables": an array indicating the model origin of each sample
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
                    f"Incompatible simulator output. "
                    f"The following keys will be dropped: {', '.join(sorted(all_keys - common_keys))}."
                )
            elif self.key_conflicts == "fill":
                logging.info(
                    f"Incompatible simulator output. "
                    f"Attempting to replace keys: {', '.join(sorted(all_keys - common_keys))}, where missing "
                    f"with value {self.fill_value}."
                )

        return keys, all_keys, common_keys, missing_keys
