from bayesflow.utils.serialization import serializable, serialize

from .transform import Transform


@serializable("bayesflow.adapters")
class Rename(Transform):
    """
    Transform to rename keys in data dictionary. Useful to rename variables to match those required by
    approximator. This transform can only rename one variable at a time.

    Parameters
    ----------
    from_key : str
        Variable name that should be renamed
    to_key : str
        New variable name

    Examples
    --------
    >>> adapter = (
        bf.adapters.Adapter()
        # rename the variables to match the required approximator inputs
        .rename("theta", "inference_variables")
        .rename("x", "inference_conditions")
    )
    """

    def __init__(self, from_key: str, to_key: str):
        super().__init__()
        self.from_key = from_key
        self.to_key = to_key

    def get_config(self) -> dict:
        return serialize({"from_key": self.from_key, "to_key": self.to_key})

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

    def extra_repr(self) -> str:
        return f"{self.from_key!r} -> {self.to_key!r}"

    def log_det_jac(self, data: dict[str, any], log_det_jac: dict[str, any], inverse: bool = False, **kwargs):
        return self.inverse(data=log_det_jac) if inverse else self.forward(data=log_det_jac, strict=False)
