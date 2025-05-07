from functools import singledispatch
from bayesflow.experimental.noise_schedules import NoiseSchedule, CosineNoiseSchedule, EDMNoiseSchedule


@singledispatch
def find_noise_schedule(arg, *args, **kwargs) -> NoiseSchedule:
    raise TypeError(f"Unknown noise schedule: {arg!r}")


@find_noise_schedule.register
def _(name: str, *args, **kwargs) -> NoiseSchedule:
    match name.lower():
        case "cosine":
            return CosineNoiseSchedule()
        case "edm":
            return EDMNoiseSchedule()
        case other:
            raise ValueError(f"Unsupported noise schedule name: '{other}'.")


@find_noise_schedule.register
def _(schedule: NoiseSchedule, *args, **kwargs) -> NoiseSchedule:
    return schedule
