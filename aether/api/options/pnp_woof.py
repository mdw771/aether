# Copyright © 2025 UChicago Argonne, LLC All right reserved
# Full license accessible at https://github.com//AdvancedPhotonSource/aether/blob/main/LICENSE

from dataclasses import dataclass, field
import os

from aether.api.options.pnp import ImageEditingOptions, PnPReconstructorOptions, PnPObjectOptions


@dataclass
class WoofOptions(ImageEditingOptions):
    checkpoint_path: str = None
    """Path to the checkpoint file (*.pth) of the model."""
    
    config_path: str = None
    """Path to the config file (*.yaml) of the model."""

    def check(self, *args, **kwargs) -> None:
        res = super().check(*args, **kwargs)
        
        if not self.checkpoint_path.endswith(".pth"):
            raise ValueError("Checkpoint must be a .pth file.")
        if not os.path.exists(self.checkpoint_path):
            raise ValueError("Checkpoint file does not exist.")
        
        if not self.config_path.endswith(".yaml") or not self.config_path.endswith(".yml"):
            raise ValueError("Config must be a .yaml file.")
        if not os.path.exists(self.config_path):
            raise ValueError("Config file does not exist.")
        
        if self.match_stats_of_prior_projected_image:
            raise NotImplementedError("Stats matching is not supported for Woof yet.")
        
        return res

@dataclass
class PnPWoofReconstructorOptions(PnPReconstructorOptions):
    prior_projection_options: WoofOptions = field(default_factory=WoofOptions)
    """Options for the prior projection."""


@dataclass
class PnPWoofObjectOptions(PnPObjectOptions):
    pass
