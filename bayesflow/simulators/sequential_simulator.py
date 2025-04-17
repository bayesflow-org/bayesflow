from collections.abc import Sequence
import numpy as np

from bayesflow.types import Shape
from bayesflow.utils.decorators import allow_batch_size

from .simulator import Simulator


class SequentialSimulator(Simulator):
    """Combines multiple simulators into one, sequentially."""

    def __init__(self, simulators: Sequence[Simulator], expand_outputs: bool = True, replace_inputs: bool = True):
        """
        Initialize a SequentialSimulator.

        Parameters
        ----------
        simulators : Sequence[Simulator]
            A sequence of simulator instances to be executed sequentially. Each simulator should
            return dictionary outputs and may depend on outputs from previous simulators.
        expand_outputs : bool, optional
            If True, 1D output arrays are expanded with an additional dimension at the end.
            Default is True.
        replace_inputs : bool, optional
            If True, **kwargs are auto-batched and replace simulator outputs.
        """

        self.simulators = simulators
        self.expand_outputs = expand_outputs
        self.replace_inputs = replace_inputs

    @allow_batch_size
    def sample(self, batch_shape: Shape, **kwargs) -> dict[str, np.ndarray]:
        """
        Sample sequentially from the internal simulator.

        Parameters
        ----------
        batch_shape : Shape
            The shape of the batch to sample. Typically, a tuple indicating the number of samples,
            but it also accepts an int.
        **kwargs
            Additional keyword arguments passed to each simulator. These may include previously
            sampled outputs used as inputs for subsequent simulators.

        Returns
        -------
        data : dict of str to np.ndarray
            A dictionary containing the combined outputs from all simulators. Keys are output names
            and values are sampled arrays. If `expand_outputs` is True, 1D arrays are expanded to
            have shape (..., 1).
        """

        data = {}
        for simulator in self.simulators:
            data |= simulator.sample(batch_shape, **(kwargs | data))

            if self.replace_inputs:
                common_keys = set(data.keys()) & set(kwargs.keys())
                for key in common_keys:
                    value = kwargs.pop(key)
                    if isinstance(data[key], np.ndarray):
                        value = np.broadcast_to(value, data[key].shape)
                    data[key] = value

        if self.expand_outputs:
            data = {
                key: np.expand_dims(value, axis=-1) if np.ndim(value) == 1 else value for key, value in data.items()
            }

        return data
