from collections.abc import Sequence
import numpy as np

from bayesflow.types import Shape
from bayesflow.utils import tree_stack
from bayesflow.utils.decorators import allow_batch_size

from bayesflow.utils import numpy_utils as npu

from .simulator import Simulator


class ModelComparisonSimulator(Simulator):
    """Wraps a sequence of simulators for use with a model comparison approximator."""

    def __init__(
        self,
        simulators: Sequence[Simulator],
        p: Sequence[float] = None,
        logits: Sequence[float] = None,
        use_mixed_batches: bool = False,
    ):
        self.simulators = simulators

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
        if not self.use_mixed_batches:
            # draw one model index for the whole batch (faster)
            model_index = np.random.choice(len(self.simulators), p=npu.softmax(self.logits))

            simulator = self.simulators[model_index]
            data = simulator.sample(batch_shape)

            model_indices = np.full(batch_shape, model_index, dtype="int32")
        else:
            # draw a model index for each sample in the batch (slower)
            model_indices = np.random.choice(len(self.simulators), p=npu.softmax(self.logits), size=batch_shape)

            data = np.empty(batch_shape, dtype="object")

            for index in np.ndindex(batch_shape):
                simulator = self.simulators[int(model_indices[index])]
                data[index] = simulator.sample(())

            data = data.flatten().tolist()
            data = tree_stack(data, axis=0, numpy=True)

            # restore batch shape
            data = {key: np.reshape(value, batch_shape + np.shape(value)[1:]) for key, value in data.items()}

        model_indices = npu.one_hot(model_indices, len(self.simulators))

        return data | {"model_indices": model_indices}
