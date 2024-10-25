import os
import pickle


def deserialize_type(config: dict) -> type:
    module = __import__(config["module"], fromlist=[config["name"]])
    return getattr(module, config["name"])


def format_bytes(b: int, precision: int = 2, si: bool = False) -> str:
    """
    Format a number of bytes as a human-readable string in the format '{value} {prefix}{unit}'.

    :param b: The number of bytes to format.
    :param precision: The display precision.
    :param si: Determines whether to use SI decimal or binary prefixes.

    Examples:
        >>> format_bytes(1024)
        '1.00 KiB'
        >>> format_bytes(1024, si=True)
        '1.02 kB'
    """
    if si:
        div = 1000
        prefixes = ["", "k", "M", "G", "T", "P", "E", "Z", "Y", "R", "Q"]
    else:
        div = 1024
        prefixes = ["", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi", "Yi", "Ri", "Qi"]

    value = float(b)
    idx = 0
    while value >= div and idx < len(prefixes) - 1:
        value /= div
        idx += 1

    prefix = prefixes[idx]

    return f"{value:.{precision}f} {prefix}B"


def parse_bytes(s: str) -> int:
    """
    Parse a string in the format '{value} {prefix}{unit}' and return the number of bytes,
    flooring to the nearest integer.

    The parsing is case-sensitive. E.g., uppercase 'K' will *not* be recognized as 'kilo',
    and lowercase 'b' will be recognized as 'bit' rather than 'byte'.

    Examples:
        >>> parse_bytes("8 GB")  # 8 Gigabyte
        8000000000
        >>> parse_bytes("32 kiB")  # 32 Kibibyte
        32768
        >>> parse_bytes("1 Tb")  # 1 Terrabit
        125000000000
        >>> parse_bytes("2.5 kB")  # 2.5 Kilobyte
        2500
        >>> parse_bytes("1e9 B")  # 10^9 Bytes
        1000000000
    """
    s = s.strip()
    if s.count(" ") != 1:
        raise ValueError(
            "Cannot parse bytes from string without exactly one space separator. "
            "Expected format: '{value} {prefix}{unit}'."
            "Example: '8 GiB'."
        )

    value, unit = s.split(" ")

    value = float(value)
    prefix, suffix = unit[:-1], unit[-1]

    if "i" not in prefix:
        # SI
        prefixes = ["", "k", "M", "G", "T", "P", "E", "Z", "Y", "R", "Q"]
        factor = 1000 ** prefixes.index(prefix)
    else:
        prefixes = ["", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi", "Yi", "Ri", "Qi"]
        factor = 1024 ** prefixes.index(prefix)

    result = int(value * factor)

    if suffix == "b":
        result //= 8

    return result


def pickle_load(filepath: os.PathLike) -> any:
    """Small utility to safely load a pickle object."""

    with open(filepath, "rb+") as f:
        return pickle.load(f)


def serialize_type(obj: any) -> dict:
    if not isinstance(obj, type):
        obj = type(obj)

    return {"module": obj.__module__, "name": obj.__qualname__}
