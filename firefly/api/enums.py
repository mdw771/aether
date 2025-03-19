from enum import StrEnum, auto


class NoiseSchedulers(StrEnum):
    DDPMScheduler = "DDPMScheduler"
    DDIMScheduler = "DDIMScheduler"
    EulerDiscreteScheduler = "EulerDiscreteScheduler"
