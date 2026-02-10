from .transform import Transform
from bayesflow.utils.serialization import deserialize, serializable, serialize


@serializable("bayesflow.adapters")
class Ungroup(Transform):
    def __init__(self, key: str, prefix: str = ""):
        """
        Ungroups the the variables in `key` from a dictionary into individual entries. Most transforms do
        not support nested structures, so this can be used to flatten a nested structure.
        It can later on be reassembled using the :py:class:`bayesflow.adapters.transforms.Group` transform.

        Parameters
        ----------
        key : str
            The name of the variable to ungroup. The variable has to be a dictionary.
        prefix : str, optional
            An optional common prefix that will be added to the ungrouped variable names. This can be necessary
            to avoid duplicate names.
        """
        super().__init__()
        self.key = key
        self.prefix = prefix
        self._ungrouped = None

    def get_config(self) -> dict:
        return serialize({"key": self.key, "prefix": self.prefix, "_ungrouped": self._ungrouped})

    @classmethod
    def from_config(cls, config: dict, custom_objects=None):
        config = deserialize(config, custom_objects)
        _ungrouped = config.pop("_ungrouped")
        transform = cls(**config)
        transform._ungrouped = _ungrouped
        return transform

    def forward(self, data: dict[str, any], *, strict: bool = True, **kwargs) -> dict[str, any]:
        data = data.copy()

        if self.key not in data and strict:
            raise KeyError(f"Missing key: {self.key!r}")
        elif self.key not in data:
            return data

        ungrouped = []
        for k, v in data.pop(self.key).items():
            new_key = f"{self.prefix}{k}"
            if new_key in data:
                raise ValueError(
                    f"Encountered duplicate key during ungrouping: '{new_key}'."
                    " Use `prefix` to specify a unique prefix that is added to the key"
                )
            ungrouped.append(new_key)
            data[new_key] = v
        if self._ungrouped is None:
            self._ungrouped = sorted(ungrouped)
        else:
            self._ungrouped = sorted(list(set(self._ungrouped + ungrouped)))

        return data

    def inverse(self, data: dict[str, any], *, strict: bool = False, **kwargs) -> dict[str, any]:
        data = data.copy()

        data[self.key] = {}
        for key in self._ungrouped:
            if key not in data:
                if strict:
                    raise KeyError(f"Missing key: {key!r}")
            else:
                recovered_key = key[len(self.prefix) :]
                data[self.key][recovered_key] = data.pop(key)

        return data

    def log_det_jac(
        self,
        data: dict[str, any],
        log_det_jac: dict[str, any],
        inverse: bool = False,
        **kwargs,
    ):
        return self.inverse(data=log_det_jac) if inverse else self.forward(data=log_det_jac, strict=False)
