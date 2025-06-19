from enum import StrEnum, auto


class NoiseSchedulers(StrEnum):
    DDPMScheduler = "DDPMScheduler"
    DDIMScheduler = "DDIMScheduler"
    EulerDiscreteScheduler = "EulerDiscreteScheduler"


class PhysicalGuidanceMethods(StrEnum):
    SCORE = auto()
    RESAMPLE = auto()
