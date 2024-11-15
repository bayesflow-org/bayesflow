from collections.abc import Sequence
import keras
import numpy as np

from bayesflow.types import Shape
from bayesflow.utils.decorators import allow_batch_size

from .simulator import Simulator


class HierarchicalSimulator(Simulator):
    def __init__(self, hierarchy: Sequence[Simulator]):
        self.hierarchy = hierarchy

    @allow_batch_size
    def sample(self, batch_shape: Shape, **kwargs) -> dict[str, np.ndarray]:
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
