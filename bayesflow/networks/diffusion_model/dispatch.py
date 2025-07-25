from functools import singledispatch

from .schedules.noise_schedule import NoiseSchedule


@singledispatch
def find_noise_schedule(arg, *args, **kwargs):
    raise TypeError(f"Not a noise schedule: {arg!r}. Please pass an object of type 'NoiseSchedule'.")


@find_noise_schedule.register
def _(noise_schedule: NoiseSchedule):
    return noise_schedule


@find_noise_schedule.register
def _(name: str, *args, **kwargs):
    match name.lower():
        case "cosine":
            from .schedules import CosineNoiseSchedule

            return CosineNoiseSchedule(*args, **kwargs)
        case "edm":
            from .schedules import EDMNoiseSchedule

            return EDMNoiseSchedule(*args, **kwargs)
        case other:
            raise ValueError(f"Unsupported noise schedule name: '{other}'.")


@find_noise_schedule.register
def _(cls: type, *args, **kwargs):
    if issubclass(cls, NoiseSchedule):
        return cls(*args, **kwargs)
    raise TypeError(f"Expected subclass of NoiseSchedule, got {cls}")
