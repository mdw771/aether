from dataclasses import dataclass, field

import ptychi.api as pcapi

import aether.api.enums as enums
from aether.api.options.guided_sampling import (
    GuidedDiffusionOptions, 
    GuidedDiffusionObjectOptions, 
    GuidedDiffusionProbeOptions, 
    GuidedDiffusionProbePositionOptions, 
    GuidedDiffusionOPRModeWeightsOptions,
    GuidedDiffusionReconstructorOptions,
)


@dataclass
class LatentDPSObjectOptions(GuidedDiffusionObjectOptions):
    pass


@dataclass
class LatentDPSProbeOptions(GuidedDiffusionProbeOptions):
    pass


@dataclass
class LatentDPSProbePositionOptions(GuidedDiffusionProbePositionOptions):
    pass


@dataclass
class LatentDPSOPRModeWeightsOptions(GuidedDiffusionOPRModeWeightsOptions):
    pass


@dataclass
class LatentDPSReconstructorOptions(GuidedDiffusionReconstructorOptions):
    pass


@dataclass
class LatentDPSOptions(GuidedDiffusionOptions):
    
    reconstructor_options: LatentDPSReconstructorOptions = field(default_factory=LatentDPSReconstructorOptions)
    
    object_options: LatentDPSObjectOptions = field(default_factory=LatentDPSObjectOptions)
    
    probe_options: LatentDPSProbeOptions = field(default_factory=LatentDPSProbeOptions)
    
    probe_position_options: LatentDPSProbePositionOptions = field(default_factory=LatentDPSProbePositionOptions)
    
    opr_mode_weight_options: LatentDPSOPRModeWeightsOptions = field(default_factory=LatentDPSOPRModeWeightsOptions)
