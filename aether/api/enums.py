# Copyright © 2025 UChicago Argonne, LLC All right reserved
# Full license accessible at https://github.com//AdvancedPhotonSource/aether/blob/main/LICENSE

from enum import StrEnum, auto


class NoiseSchedulers(StrEnum):
    DDPMScheduler = "DDPMScheduler"
    DDIMScheduler = "DDIMScheduler"
    EulerDiscreteScheduler = "EulerDiscreteScheduler"


class PhysicalGuidanceMethods(StrEnum):
    SCORE = auto()
    RESAMPLE = auto()
