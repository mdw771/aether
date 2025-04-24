from dataclasses import dataclass, field

import ptychi.api as pcapi

import firefly.api.enums as enums
from firefly.api.options.guided_sampling import (
    GuidedDiffusionOptions, 
    GuidedDiffusionObjectOptions, 
    GuidedDiffusionProbeOptions, 
    GuidedDiffusionProbePositionOptions, 
    GuidedDiffusionOPRModeWeightsOptions,
    GuidedDiffusionReconstructorOptions,
)


@dataclass
class ADMMDiffObjectOptions(GuidedDiffusionObjectOptions):
    pass


@dataclass
class ADMMDiffProbeOptions(GuidedDiffusionProbeOptions):
    pass


@dataclass
class ADMMDiffProbePositionOptions(GuidedDiffusionProbePositionOptions):
    pass


@dataclass
class ADMMDiffOPRModeWeightsOptions(GuidedDiffusionOPRModeWeightsOptions):
    pass


@dataclass
class ADMMDiffReconstructorOptions(GuidedDiffusionReconstructorOptions):
    pass


@dataclass
class ADMMDiffOptions(GuidedDiffusionOptions):
    
    reconstructor_options: ADMMDiffReconstructorOptions = field(default_factory=ADMMDiffReconstructorOptions)
    
    object_options: ADMMDiffObjectOptions = field(default_factory=ADMMDiffObjectOptions)
    
    probe_options: ADMMDiffProbeOptions = field(default_factory=ADMMDiffProbeOptions)
    
    probe_position_options: ADMMDiffProbePositionOptions = field(default_factory=ADMMDiffProbePositionOptions)
    
    opr_mode_weight_options: ADMMDiffOPRModeWeightsOptions = field(default_factory=ADMMDiffOPRModeWeightsOptions)
