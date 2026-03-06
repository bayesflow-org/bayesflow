from collections.abc import Callable, Mapping, Sequence


def apply_augmentations(batch, augmentations):
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
