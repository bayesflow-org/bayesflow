from collections.abc import Callable, Sequence
import numpy as np

from bayesflow.types import Shape
from bayesflow.utils import tree_concatenate
from bayesflow.utils.decorators import allow_batch_size

from bayesflow.utils import numpy_utils as npu

from types import FunctionType

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
            If True, samples in a batch are drawn from different models. If False, the entire batch
            is drawn from a single model chosen according to the model probabilities. Default is True.
        shared_simulator : Simulator or Callable, optional
            A shared simulator whose outputs are passed to all model simulators. If a function is
            provided, it is wrapped in a `LambdaSimulator` with batching enabled.
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
            sims = tree_concatenate(sims, numpy=True)
            data |= sims

            model_indices = np.repeat(np.eye(num_models, dtype="int32"), model_counts, axis=0)

        # draw one model index for the whole batch (faster)
        else:
            model_index = np.random.choice(num_models, p=softmax_logits)

            data = self.simulators[model_index].sample(batch_shape, **(kwargs | data))
            model_indices = npu.one_hot(np.full(batch_shape, model_index, dtype="int32"), num_models)

        return data | {"model_indices": model_indices}
