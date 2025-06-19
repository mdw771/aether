import aether.api.enums as enums
import diffusers.schedulers as schedulers


def get_noise_scheduler(scheduler: enums.NoiseSchedulers):
    return getattr(schedulers, scheduler)
