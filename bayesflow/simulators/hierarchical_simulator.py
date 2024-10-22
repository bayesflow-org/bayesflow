from collections.abc import Sequence
import keras
import numpy as np

from bayesflow.types import ShapeLike

from .simulator import Simulator
from bayesflow.utils import validate_shape


class HierarchicalSimulator(Simulator):
    def __init__(self, hierarchy: Sequence[Simulator]):
        self.hierarchy = hierarchy

    def sample(self, batch_shape: ShapeLike, **kwargs) -> dict[str, np.ndarray]:
        batch_shape = validate_shape(batch_shape)

        input_data = {}
        output_data = {}

        for level in range(len(self.hierarchy)):
            # repeat input data for the next level
            def repeat_level(x):
                return np.repeat(x, batch_shape[level], axis=0)

            input_data = keras.tree.map_structure(repeat_level, input_data)

            # query the simulator flat at the current level
            simulator = self.hierarchy[level]
            query_shape = (np.prod(batch_shape[: level + 1]),)
            data = simulator.sample(query_shape, **(kwargs | input_data))

            # input data needs to have a flat batch shape
            input_data |= data

            # output data needs the restored batch shape
            def restore_batch_shape(x):
                return np.reshape(x, batch_shape[: level + 1] + x.shape[1:])

            data = keras.tree.map_structure(restore_batch_shape, data)
            output_data |= data

        return output_data
