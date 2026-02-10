from collections.abc import Sequence
import keras
import numpy as np

from bayesflow.types import Shape
from bayesflow.utils.decorators import allow_batch_size

from .simulator import Simulator


class HierarchicalSimulator(Simulator):
    def __init__(self, hierarchy: Sequence[Simulator]):
        """
        Initialize the hierarchical simulator with a sequence of simulators.

        Parameters
        ----------
        hierarchy : Sequence[Simulator]
            A sequence of simulator instances representing each level of the hierarchy.
            Each level's output is used as input for the next, with increasing batch dimensions.
        """
        self.hierarchy = hierarchy

    @allow_batch_size
    def sample(self, batch_shape: Shape, **kwargs) -> dict[str, np.ndarray]:
        """
        Sample from a hierarchy of simulators.

        Parameters
        ----------
        batch_shape : Shape
            A tuple where each element specifies the number of samples at the corresponding level
            of the hierarchy. The total batch size increases multiplicatively through the levels.
        **kwargs
            Additional keyword arguments passed to each simulator. These are combined with outputs
            from previous levels and repeated appropriately.

        Returns
        -------
        output_data : dict of str to np.ndarray
            A dictionary containing the outputs from the entire hierarchy. Outputs are reshaped to
            match the hierarchical batch shape, i.e., with shape equal to `batch_shape + original_shape`.
        """

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
