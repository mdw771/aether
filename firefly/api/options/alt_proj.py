from dataclasses import dataclass, field
from typing import Optional

import ptychi.api as pcapi
import firefly.api.enums as enums


@dataclass
class AlternatingProjectionReconstructorOptions(pcapi.options.ad_ptychography.AutodiffPtychographyReconstructorOptions):
    prompt: str = ""
    """The prompt to use for the guided sampling."""
    
    negative_prompt: Optional[str] = None
    """The negative prompt to use for the guided sampling."""
    
    text_guidance_scale: float = 4.5
    """The guidance scale to use for the guided sampling."""
    
    physical_guidance_scale: float = 0.1
    """The guidance scale to use for the physical guidance."""
    
    prior_strength: float = 0.5
    """The strength of the prior during the prior projection step. This should
    be a value between 0 and 1. 0 means no prior is used. 1 means the prior
    is at full power and the input image is effectively ignored.
    """
    
    loss_function: pcapi.enums.LossFunctions = pcapi.enums.LossFunctions.MSE_SQRT
    """The loss function to calculate physical guidance."""
    
    num_inference_steps: int = 50
    """The number of inference steps to use for the guided sampling."""
    
    model_path: str = "stabilityai/stable-diffusion-xl-base-1.0"
    """The path to the model to use for the guided sampling."""
    
    noise_scheduler: enums.NoiseSchedulers = None
    """The noise scheduler to use for the guided sampling."""
    
    num_epochs: int = 20
    """The number of outer epochs."""
    
    num_data_projection_epochs: int = 20
    """The number of epochs for data projection."""
    
    proximal_penalty: float = 1.0
    """The penalty for the proximal term in calculating the projections. This is the
    `tau` in https://arxiv.org/abs/1704.02712. ADMM reduces to alternating projection 
    when `proximal_penalty == 0` and `update_relaxation == 1`.
    """
    
    update_relaxation: float = 1.0
    """The relaxation factor in ADMM. This is the `gamma` in https://arxiv.org/abs/1704.02712.
    ADMM reduces to alternating projection when `proximal_penalty == 0` and 
    `update_relaxation == 1`.
    """
    
    use_prior_projected_data_as_final_result: bool = True
    """If True, the object data at the end of the reconstruction is set to the data after
    prior projection (v). Otherwise, it will be set to the data after data projection (x),
    i.e., the step before prior projection.
    """
    
    forward_model_class: pcapi.enums.ForwardModels = pcapi.enums.ForwardModels.PLANAR_PTYCHOGRAPHY
    """The forward model to use for physical guidance"""
    
    def get_reconstructor_type(self):
        return "alternating_projection"
    
    def check(self, options: pcapi.options.task_options.PtychographyTaskOptions) -> None:
        super().check(options)


@dataclass
class AlternatingProjectionObjectOptions(pcapi.options.base.ObjectOptions):
    pass


@dataclass
class AlternatingProjectionProbeOptions(pcapi.options.base.ProbeOptions):
    pass


@dataclass
class AlternatingProjectionProbePositionOptions(pcapi.options.base.ProbePositionOptions):
    pass


@dataclass
class AlternatingProjectionOPRModeWeightsOptions(pcapi.options.base.OPRModeWeightsOptions):
    pass


@dataclass
class AlternatingProjectionOptions(pcapi.options.task.PtychographyTaskOptions):
    
    reconstructor_options: AlternatingProjectionReconstructorOptions = field(default_factory=AlternatingProjectionReconstructorOptions)
    
    object_options: AlternatingProjectionObjectOptions = field(default_factory=AlternatingProjectionObjectOptions)
    
    probe_options: AlternatingProjectionProbeOptions = field(default_factory=AlternatingProjectionProbeOptions)
    
    probe_position_options: AlternatingProjectionProbePositionOptions = field(default_factory=AlternatingProjectionProbePositionOptions)
    
    opr_mode_weight_options: AlternatingProjectionOPRModeWeightsOptions = field(default_factory=AlternatingProjectionOPRModeWeightsOptions)
    