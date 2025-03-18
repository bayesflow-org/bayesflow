from collections.abc import Sequence
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
        shared_simulator: Simulator | FunctionType = None,
    ):
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
        data = {}
        if self.shared_simulator:
            data |= self.shared_simulator.sample(batch_shape, **kwargs)

        if not self.use_mixed_batches:
            # draw one model index for the whole batch (faster)
            model_index = np.random.choice(len(self.simulators), p=npu.softmax(self.logits))

            simulator = self.simulators[model_index]
            data = simulator.sample(batch_shape, **(kwargs | data))

            model_indices = np.full(batch_shape, model_index, dtype="int32")
            model_indices = npu.one_hot(model_indices, len(self.simulators))
        else:
            # generate data randomly from each model (slower)
            model_counts = np.random.multinomial(n=batch_shape[0], pvals=npu.softmax(self.logits))

            sims = []
            for n, simulator in zip(model_counts, self.simulators):
                if n == 0:
                    continue
                sim = simulator.sample(n, **(kwargs | data))
                sims.append(sim)

            sims = tree_concatenate(sims, numpy=True)
            data |= sims

            model_indices = np.eye(len(self.simulators), dtype="int32")
            model_indices = np.repeat(model_indices, model_counts, axis=0)

        return data | {"model_indices": model_indices}
