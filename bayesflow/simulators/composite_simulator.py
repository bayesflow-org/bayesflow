from collections.abc import Sequence
import numpy as np

from bayesflow.types import ShapeLike
from bayesflow.utils import validate_shape

from .simulator import Simulator


class CompositeSimulator(Simulator):
    """Combines multiple simulators into one, sequentially."""

    def __init__(self, simulators: Sequence[Simulator], expand_outputs: bool = False):
        self.simulators = simulators
        self.expand_outputs = expand_outputs

    def sample(self, batch_shape: ShapeLike, **kwargs) -> dict[str, np.ndarray]:
        batch_shape = validate_shape(batch_shape)

        data = {}
        for simulator in self.simulators:
            data |= simulator.sample(batch_shape, **(kwargs | data))

        if self.expand_outputs:
            data = {
                key: np.expand_dims(value, axis=-1) if np.ndim(value) == 1 else value for key, value in data.items()
            }

        return data
