from bayesflow.types import Tensor
from bayesflow.utils.serialization import serializable, deserialize


@serializable("bayesflow.adapters")
class Transform:
    """
    Base class on which other transforms are based
    """

    def __call__(self, data: dict[str, Tensor], *, inverse: bool = False, **kwargs) -> dict[str, Tensor]:
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

    def forward(self, data: dict[str, Tensor], **kwargs) -> dict[str, Tensor]:
        raise NotImplementedError

    def inverse(self, data: dict[str, Tensor], **kwargs) -> dict[str, Tensor]:
        raise NotImplementedError

    def extra_repr(self) -> str:
        return ""

    def log_det_jac(
        self, data: dict[str, Tensor], log_det_jac: dict[str, Tensor], inverse: bool = False, **kwargs
    ) -> dict[str, Tensor]:
        return log_det_jac
