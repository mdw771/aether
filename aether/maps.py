# Copyright © 2025 UChicago Argonne, LLC All right reserved
# Full license accessible at https://github.com//AdvancedPhotonSource/aether/blob/main/LICENSE

import aether.api.enums as enums
import diffusers.schedulers as schedulers


def get_noise_scheduler(scheduler: enums.NoiseSchedulers):
    return getattr(schedulers, scheduler)
