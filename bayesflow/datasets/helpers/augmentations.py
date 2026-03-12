from collections.abc import Callable, Mapping, Sequence


def apply_augmentations(batch, augmentations):
    """Apply augmentations to a batch of data.

    This function supports multiple augmentation patterns:
    - None: Returns the batch unchanged
    - Mapping (dict-like): Applies each function to the corresponding key's data
    - Sequence (list/tuple): Applies each function sequentially to the batch
    - Callable: Applies the function directly to the batch

    Args:
        batch: The data batch to augment. Can be any data structure compatible
            with the augmentation functions.
        augmentations: The augmentation(s) to apply. Can be:
            - None: No augmentations applied
            - Callable: A single function to apply to the batch
            - Mapping: A dict mapping keys to functions that transform batch[key]
            - Sequence: A list/tuple of functions to apply sequentially to the batch

    Returns:
        The augmented batch with the same structure as the input batch.

    Raises:
        RuntimeError: If augmentations is of an unsupported type.

    Examples:
        # No augmentation
        batch = apply_augmentations(batch, None)

        # Single function
        batch = apply_augmentations(batch, lambda x: x * 2)

        # Mapping (key-based augmentation)
        batch = apply_augmentations(batch, {
            'features': normalize,
            'labels': one_hot_encode
        })

        # Sequence (sequential augmentation)
        batch = apply_augmentations(batch, [normalize, standardize, augment])
    """
    match augmentations:
        case None:
            return batch

        case Mapping() as aug:
            for key, fn in aug.items():
                batch[key] = fn(batch[key])
            return batch

        case Sequence() as augs if not isinstance(augs, (str, bytes)):
            for fn in augs:
                batch = fn(batch)
            return batch

        case Callable() as fn:
            return fn(batch)

        case _:
            raise RuntimeError(f"Could not apply augmentations of type {type(augmentations)}.")
