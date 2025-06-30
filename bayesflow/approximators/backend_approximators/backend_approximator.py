import keras

from bayesflow.utils import filter_kwargs


match keras.backend.backend():
    case "jax":
        from .jax_approximator import JAXApproximator as BaseBackendApproximator
    case "tensorflow":
        from .tensorflow_approximator import TensorFlowApproximator as BaseBackendApproximator
    case "torch":
        from .torch_approximator import TorchApproximator as BaseBackendApproximator
    case other:
        raise ValueError(f"Backend '{other}' is not supported.")


class BackendApproximator(BaseBackendApproximator):
    # noinspection PyMethodOverriding
    def fit(self, *, dataset: keras.utils.PyDataset, **kwargs):
        return super().fit(x=dataset, y=None, **filter_kwargs(kwargs, super().fit))
