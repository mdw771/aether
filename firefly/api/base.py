from dataclasses import dataclass, field

import ptychi.api as pcapi
import firefly.api.enums as enums


@dataclass
class GuidedDiffusionReconstructorOptions(pcapi.options.ad_ptychography.AutodiffPtychographyReconstructorOptions):
    prompt: str = ""
    """The prompt to use for the guided sampling."""
    
    text_guidance_scale: float = 4.5
    """The guidance scale to use for the guided sampling."""
    
    physical_guidance_scale: float = 0.1
    """The guidance scale to use for the physical guidance."""
    
    loss_function: pcapi.enums.LossFunctions = pcapi.enums.LossFunctions.MSE_SQRT
    """The loss function to calculate physical guidance."""
    
    num_inference_steps: int = 50
    """The number of inference steps to use for the guided sampling."""
    
    time_travel_steps: int = 10
    """The number of steps to travel back in time."""
    
    time_travel_interval: int = 1
    """The interval at which to travel back in time."""
    
    model_path: str = "stabilityai/stable-diffusion-xl-base-1.0"
    """The path to the model to use for the guided sampling."""
    
    noise_scheduler: enums.NoiseSchedulers = enums.NoiseSchedulers.DDPMScheduler
    """The noise scheduler to use for the guided sampling."""
    num_epochs: int = 1
    """The number of epochs. An epoch here refers to one generation pass, which
    may be multiple steps of the guided sampling. Usually 1 is enough, but it can
    be increased to enable multi-pass generation.
    """
    
    forward_model_class: pcapi.enums.ForwardModels = pcapi.enums.ForwardModels.PLANAR_PTYCHOGRAPHY
    """The forward model to use for physical guidance"""
    
    def get_reconstructor_type(self):
        return "guided_diffusion"


@dataclass
class GuidedDiffusionObjectOptions(pcapi.options.base.ObjectOptions):
    pass


@dataclass
class GuidedDiffusionProbeOptions(pcapi.options.base.ProbeOptions):
    pass


@dataclass
class GuidedDiffusionProbePositionOptions(pcapi.options.base.ProbePositionOptions):
    pass


@dataclass
class GuidedDiffusionOPRModeWeightsOptions(pcapi.options.base.OPRModeWeightsOptions):
    pass


@dataclass
class GuidedDiffusionOptions(pcapi.options.task.PtychographyTaskOptions):
    
    reconstructor_options: GuidedDiffusionReconstructorOptions = field(default_factory=GuidedDiffusionReconstructorOptions)
    
    object_options: GuidedDiffusionObjectOptions = field(default_factory=GuidedDiffusionObjectOptions)
    
    probe_options: GuidedDiffusionProbeOptions = field(default_factory=GuidedDiffusionProbeOptions)
    
    probe_position_options: GuidedDiffusionProbePositionOptions = field(default_factory=GuidedDiffusionProbePositionOptions)
    
    opr_mode_weight_options: GuidedDiffusionOPRModeWeightsOptions = field(default_factory=GuidedDiffusionOPRModeWeightsOptions)
    
