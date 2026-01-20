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

    def sample(self, batch_size: int, sample_shape: tuple[int] | None = None, **kwargs) -> dict[str, np.ndarray]:
        """
        Sample sequentially from the internal simulator.

        Parameters
        ----------
        batch_size : int
            The number of samples to generate.
        sample_shape : tuple of int or int, optional
            Trailing structural dimensions of each generated sample, excluding the batch and target (intrinsic)
            dimension. For example, if batch_size is `batch_size` and sample_shape is `(time, channels)`, the final
            output will be `(batch_size, time, channels, target_dim)`, where target_dim is the intrinsic dimension of
            the output.
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
            data |= simulator.sample(batch_size, sample_shape=sample_shape, **(kwargs | data))

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

    def _single_sample(self, batch_shape_ext, **kwargs) -> dict[str, np.ndarray]:
        """
        For single sample used by parallel sampling.

        Parameters
        ----------
        **kwargs
            Keyword arguments passed to simulators.

        Returns
        -------
        dict
            Single sample result.
        """
        return self.sample(batch_size=1, sample_shape=(batch_shape_ext), **kwargs)

    def sample_parallel(
        self, batch_size: int, sample_shape: tuple[int] | None = None, n_jobs: int = -1, verbose: int = 0, **kwargs
    ) -> dict[str, np.ndarray]:
        """
        Sample in parallel from the sequential simulator.

        Parameters
        ----------
        batch_size : int
            The number of samples to generate.
        sample_shape : tuple of int or int, optional
            Trailing structural dimensions of each generated sample, excluding the batch and target (intrinsic)
            dimension. For example, if batch_size is `batch_size` and sample_shape is `(time, channels)`, the final
            output will be `(batch_size, time, channels, target_dim)`, where target_dim is the intrinsic dimension of
            the output.
        n_jobs : int, optional
            Number of parallel jobs. -1 uses all available cores. Default is -1.
        verbose : int, optional
            Verbosity level for joblib. Default is 0 (no output).
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
        try:
            from joblib import Parallel, delayed
        except ImportError as e:
            raise ImportError(
                "joblib is required for parallel sampling. Please install it via 'pip install joblib'."
            ) from e

        batch_shape = (batch_size, *sample_shape) if sample_shape else (batch_size,)
        if len(batch_shape) == 0:
            raise ValueError("batch_shape must be a positive integer or a nonempty tuple")

        results = Parallel(n_jobs=n_jobs, verbose=verbose)(
            delayed(self._single_sample)(batch_shape_ext=batch_shape[1:], **kwargs) for _ in range(batch_shape[0])
        )
        return self._combine_results(results)

    @staticmethod
    def _combine_results(results: list[dict]) -> dict[str, np.ndarray]:
        """
        Combine a list of single-sample results into arrays.

        Parameters
        ----------
        results : list of dict
            List of dictionaries from individual samples.

        Returns
        -------
        dict
            Combined results with arrays.
        """
        if not results:
            return {}

        # union of all keys across results
        all_keys = set()
        for r in results:
            all_keys.update(r.keys())

        combined_data: dict[str, np.ndarray] = {}

        for key in all_keys:
            values = []
            for result in results:
                if key in result:
                    value = result[key]
                    if isinstance(value, np.ndarray) and value.shape[:1] == (1,):
                        values.append(value[0])
                    else:
                        values.append(value)
                else:
                    values.append(None)

            try:
                if all(isinstance(v, np.ndarray) for v in values):
                    combined_data[key] = np.stack(values, axis=0)
                else:
                    combined_data[key] = np.array(values, dtype=object)
            except ValueError:
                combined_data[key] = np.array(values, dtype=object)

        return combined_data
