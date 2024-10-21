from keras.saving import (
    register_keras_serializable as serializable,
)

from .transform import Transform


@serializable(package="bayesflow.data_adapters")
class Rename(Transform):
    def __init__(self, from_key: str, to_key: str):
        super().__init__()
        self.from_key = from_key
        self.to_key = to_key

    @classmethod
    def from_config(cls, config: dict, custom_objects=None) -> "Rename":
        return cls(
            from_key=config["from_key"],
            to_key=config["to_key"],
        )

    def get_config(self) -> dict:
        return {"from_key": self.from_key, "to_key": self.to_key}

    def forward(self, data: dict[str, any], *, strict: bool = True, **kwargs) -> dict[str, any]:
        data = data.copy()

        if strict and self.from_key not in data:
            raise KeyError(f"Missing key: {self.from_key!r}")
        elif self.from_key not in data:
            return data

        data[self.to_key] = data.pop(self.from_key)
        return data

    def inverse(self, data: dict[str, any], *, strict: bool = False, **kwargs) -> dict[str, any]:
        data = data.copy()

        if strict and self.to_key not in data:
            raise KeyError(f"Missing key: {self.to_key!r}")
        elif self.to_key not in data:
            return data

        data[self.from_key] = data.pop(self.to_key)
        return data
