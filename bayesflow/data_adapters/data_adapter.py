from typing import Generic, TypeVar


TRaw = TypeVar("TRaw")
TProcessed = TypeVar("TProcessed")


class DataAdapter(Generic[TRaw, TProcessed]):
    """Construct and deconstruct deep-learning ready data from and into raw data."""

    def configure(self, raw_data: TRaw) -> TProcessed:
        """Construct deep-learning ready data from raw data."""
        raise NotImplementedError

    def deconfigure(self, processed_data: TProcessed) -> TRaw:
        """Reconstruct raw data from deep-learning ready processed data.
        Note that configuration is not required to be bijective, so this method is only meant to be a 'best effort'
        attempt, and may return incomplete or different raw data.
        """
        raise NotImplementedError

    @classmethod
    def from_config(cls, config: dict, custom_objects=None) -> "DataAdapter":
        """Construct a data adapter from a configuration dictionary."""
        raise NotImplementedError

    def get_config(self) -> dict:
        """Return a configuration dictionary."""
        raise NotImplementedError
