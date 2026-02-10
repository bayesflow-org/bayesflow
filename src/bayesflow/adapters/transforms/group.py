from collections.abc import Sequence
from .transform import Transform
from bayesflow.utils.serialization import serializable, serialize


@serializable("bayesflow.adapters")
class Group(Transform):
    def __init__(self, keys: Sequence[str], into: str, prefix: str = ""):
        """Groups the given variables as a dictionary in the key `into`. As most transforms do
        not support nested structures, this should usually be the last transform.

        Parameters
        ----------
        keys : Sequence of str
            The names of the variables to group together.
        into : str
            The name of the variable to store the grouped variables in.
        prefix : str, optional
            A common prefix of the ungrouped variable names, which will be removed after grouping.

        Raises
        ------
        ValueError
            If a prefix is specified, but a provided key does not start with the prefix.
        """
        super().__init__()
        self.keys = keys
        self.into = into
        self.prefix = prefix
        for key in keys:
            if not key.startswith(prefix):
                raise ValueError(f"If prefix is specified, all keys have to start with prefix. Found '{key}'.")

    def get_config(self) -> dict:
        return serialize({"keys": self.keys, "into": self.into, "prefix": self.prefix})

    def forward(self, data: dict[str, any], *, strict: bool = True, **kwargs) -> dict[str, any]:
        data = data.copy()

        data[self.into] = data.get(self.into, {})
        for key in self.keys:
            if key not in data:
                if strict:
                    raise KeyError(f"Missing key: {key!r}")
            else:
                data[self.into][key[len(self.prefix) :]] = data.pop(key)

        return data

    def inverse(self, data: dict[str, any], *, strict: bool = False, **kwargs) -> dict[str, any]:
        data = data.copy()

        if strict and self.into not in data:
            raise KeyError(f"Missing key: {self.into!r}")
        elif self.into not in data:
            return data

        for key in self.keys:
            internal_key = key[len(self.prefix) :]
            if internal_key not in data[self.into]:
                if strict:
                    raise KeyError(f"Missing key: {internal_key!r}")
            else:
                data[key] = data[self.into].pop(internal_key)

        if len(data[self.into]) == 0:
            del data[self.into]

        return data

    def extra_repr(self) -> str:
        return f"{self.keys!r} -> {self.into!r}"

    def log_det_jac(
        self,
        data: dict[str, any],
        log_det_jac: dict[str, any],
        inverse: bool = False,
        **kwargs,
    ):
        return self.inverse(data=log_det_jac) if inverse else self.forward(data=log_det_jac, strict=False)
