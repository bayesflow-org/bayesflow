import numpy as np

from bayesflow.utils.serialization import serializable, deserialize


@serializable("bayesflow.adapters")
class Transform:
    """
    Base class on which other transforms are based
    """

    def __call__(self, data: dict[str, np.ndarray], *, inverse: bool = False, **kwargs) -> dict[str, np.ndarray]:
        if inverse:
            return self.inverse(data, **kwargs)

        return self.forward(data, **kwargs)

    def __repr__(self):
        if e := self.extra_repr():
            return f"{self.__class__.__name__}({e})"
        return self.__class__.__name__

    @classmethod
    def from_config(cls, config: dict, custom_objects=None):
        # noinspection PyArgumentList
        return cls(**deserialize(config, custom_objects=custom_objects))

    def get_config(self) -> dict:
        raise NotImplementedError

    def forward(self, data: dict[str, np.ndarray], **kwargs) -> dict[str, np.ndarray]:
        raise NotImplementedError

    def inverse(self, data: dict[str, np.ndarray], **kwargs) -> dict[str, np.ndarray]:
        raise NotImplementedError

    def extra_repr(self) -> str:
        return ""

    def log_det_jac(
        self, data: dict[str, np.ndarray], log_det_jac: dict[str, np.ndarray], inverse: bool = False, **kwargs
    ) -> dict[str, np.ndarray]:
        return log_det_jac
