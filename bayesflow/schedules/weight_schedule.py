from bayesflow.utils.serialization import serializable, serialize, deserialize
import keras.ops as ops


class WeightSchedule:
    def __init__(self, min: float = 0.0, max: float = 1.0, **kwargs):
        self.min = float(min)
        self.max = float(max)

    def __call__(self, step: int):
        raise NotImplementedError

    @classmethod
    def from_config(cls, config, custom_objects=None):
        return cls(**deserialize(config, custom_objects=custom_objects))

    def get_config(self):
        return {"min": self.min, "max": self.max}


@serializable("bayesflow.schedules")
class ConstantSchedule(WeightSchedule):
    def __init__(self, weight: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.weight = float(weight)

    def __call__(self, step: int):
        return self.weight

    def get_config(self):
        base_config = super().get_config()
        config = {"weight": self.weight}

        return base_config | serialize(config)


@serializable("bayesflow.schedules")
class LinearSchedule(WeightSchedule):
    def __init__(self, min_step: int = 0, max_step: int = 1, **kwargs):
        super().__init__(**kwargs)
        self.min_step = min_step
        self.max_step = max_step

    def __call__(self, step: int):
        step = ops.cast(step, "float32")
        s0 = ops.cast(self.min_step, "float32")
        s1 = ops.cast(self.max_step, "float32")

        # distance in steps (avoid division by 0)
        denom = ops.maximum(ops.abs(s1 - s0), 1.0)

        # progress from 0->1 as we move from min_step to max_step
        # works even if max_step < min_step
        t = (step - s0) / denom
        t = ops.clip(t, 0.0, 1.0)

        # If increasing: start=min, end=max
        w_inc = self.min + t * (self.max - self.min)

        # If decreasing: start=max, end=min
        w_dec = self.max + t * (self.min - self.max)

        # Choose based on direction (tensor-safe)
        weight = ops.where(s1 >= s0, w_inc, w_dec)
        return weight

    def get_config(self):
        base_config = super().get_config()
        config = {"min_step": self.min_step, "max_step": self.max_step}
        return base_config | serialize(config)
