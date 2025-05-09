from functools import singledispatch
from .noise_schedules import NoiseSchedule


@singledispatch
def find_noise_schedule(arg, *args, **kwargs):
    raise TypeError(f"Unknown noise schedule: {arg!r}")


@find_noise_schedule.register
def _(noise_schedule: NoiseSchedule):
    return noise_schedule


@find_noise_schedule.register
def _(name: str, *args, **kwargs):
    match name.lower():
        case "cosine":
            from .noise_schedules import CosineNoiseSchedule

            return CosineNoiseSchedule()
        case "edm":
            from .noise_schedules import EDMNoiseSchedule

            return EDMNoiseSchedule()
        case other:
            raise ValueError(f"Unsupported noise schedule name: '{other}'.")


@find_noise_schedule.register
def _(config: dict, *args, **kwargs):
    name = config.get("name", "").lower()
    params = {k: v for k, v in config.items() if k != "name"}
    match name:
        case "cosine":
            from .noise_schedules import CosineNoiseSchedule

            return CosineNoiseSchedule(**params)
        case "edm":
            from .noise_schedules import EDMNoiseSchedule

            return EDMNoiseSchedule(**params)
        case other:
            raise ValueError(f"Unsupported noise schedule config: '{other}'.")


@find_noise_schedule.register
def _(cls: type, *args, **kwargs):
    if issubclass(cls, NoiseSchedule):
        return cls(*args, **kwargs)
    raise TypeError(f"Expected subclass of NoiseSchedule, got {cls}")


@find_noise_schedule.register
def _(schedule: type, *args, **kwargs):
    return schedule
