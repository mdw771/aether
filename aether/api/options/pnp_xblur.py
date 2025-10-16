# Copyright © 2025 UChicago Argonne, LLC All right reserved
# Full license accessible at https://github.com//AdvancedPhotonSource/aether/blob/main/LICENSE

from dataclasses import dataclass

from aether.api.options.pnp import PriorProjectionOptions


@dataclass
class CWTBlurryFeatureRemovalOptions(PriorProjectionOptions):
    n_levels: int = 4
    """The number of levels of the DTCWT."""
    
    attenuation_threshold: float = 0.1
    """The threshold for the attenuation of the highpass coefficients."""
