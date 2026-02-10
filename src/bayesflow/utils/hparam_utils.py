import numpy as np

from .io import format_bytes, parse_bytes
from . import logging
from .tensor_utils import size_of


def find_batch_size(sample, memory_budget: str = "auto", min: int = 4, max: int = 1024) -> int:
    """Returns an estimation of an optimal batch size based on memory budget and sample memory footprint.

    :param sample: Any nested structure of tensors, representing a single sample.
    :param memory_budget: The maximum available memory for a single batch.
    :param min: The minimum batch size.
    :param max: The maximum batch size.
    """
    if memory_budget == "auto":
        memory_budget = find_memory_budget()
    elif isinstance(memory_budget, str):
        memory_budget = parse_bytes(memory_budget)

    # find the size of one sample
    sample_memory = size_of(sample)

    logging.info(f"Estimating memory footprint of one sample at {format_bytes(sample_memory, precision=1)}.")

    # use a conservative (low) estimate for the optimal batch size
    batch_size = memory_budget / (4 * sample_memory)

    if batch_size < 16:
        logging.warning(
            "Memory budget is very small compared to sample size. You may need to accumulate gradients over "
            f"multiple batches using `gradient_accumulation_steps` in the optimizer. We recommend accumulating "
            f"at least {int(32 / batch_size)} steps."
        )

    # limit estimate to sensible range
    batch_size = int(np.clip(batch_size, min, max))

    return batch_size


def find_memory_budget(device_type: str = None) -> int:
    """Returns an estimation of available memory in bytes for the given device type."""
    # keras utilities for device information are not very mature yet
    raise NotImplementedError("Automatic memory budget is not yet supported. Please pass an explicit value.")
