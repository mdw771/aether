from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import ptychi.api as pcapi

import firefly.api.enums as enums


@dataclass
class ResampleOptions(pcapi.options.base.Options):
    num_z_optimization_epochs: int = 50
    """The number of epochs to optimize the latent code $\hat{z}_0$."""
    
    gamma: float = 40
    """Coefficient of the schedule for $\sigma_t$, where $\sigma_t$ is calculated as
    $\sigma_t^2 = \gamma \left( \frac{1 - \bar{\alpha_{t-1}}}{\bar{\alpha_t}} \right) \left( \frac{1 - \bar{\alpha_t}}{\bar{\alpha_{t-1}}} \right)$.
    """
    
    optimizer: pcapi.enums.Optimizers = pcapi.enums.Optimizers.ADAM
    """The optimizer."""
    
    step_size: float = 1e-3
    """The learning rate."""
    
    frequency_loss_weight: float = 0
    """Weight of the loss term that promotes power in high-frequency regime."""
    
    optimization_space_dividing_point: float = 0.8
    """The fraction of the total number of inference steps that divides the optimization space
    in ReSample's physical guidance step. For example, if the total number of inference steps is 
    500, and this value is 0.8, then physical guidance is done by optimizing the decoded pixel-space
    image before the 400th step, and optimizing the latent after the 400th step.
    """


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
    
    time_travel_plan: pcapi.OptimizationPlan = field(default_factory=lambda: pcapi.OptimizationPlan(start=np.inf, stride=np.inf))
    """The scheduling plan for time travel where one can set the start, stop and interval
    in terms of the denoising step."""
    
    model_path: str = "stabilityai/stable-diffusion-xl-base-1.0"
    """The path to the model to use for the guided sampling."""
    
    noise_scheduler: enums.NoiseSchedulers = enums.NoiseSchedulers.DDPMScheduler
    """The noise scheduler to use for the guided sampling."""
    
    physical_guidance_method: enums.PhysicalGuidanceMethods = enums.PhysicalGuidanceMethods.RESAMPLE
    """The method to use for physical guidance.
    
    - SCORE: Compute the score function of the physics model as the gradient of the data fidelity
      loss, and add the gradient to the predicted noise.
    
    - RESAMPLE: Use the ReSample method in https://arxiv.org/abs/2307.08123, where the estimated
      latent at timestep 0 is minimized as a subproblem and then stochastically resampled back
      to timestep t.
    """
    
    physical_guidance_plan: pcapi.OptimizationPlan = field(default_factory=pcapi.OptimizationPlan)
    """The scheduling plan for physical guidance where one can set the start, stop and interval
    in terms of the denoising step."""
    
    resample_options: ResampleOptions = field(default_factory=ResampleOptions)
    """Options for the ReSample method."""
    
    proximal_penalty: float = 0
    """The proximal penalty for ADMM. Set to 0 to disable ADMM."""
    
    num_epochs: int = 1
    """The number of epochs. An epoch here refers to one generation pass, which
    may be multiple steps of the guided sampling. Usually 1 is enough, but it can
    be increased to enable multi-pass generation.
    """
    
    forward_model_class: pcapi.enums.ForwardModels = pcapi.enums.ForwardModels.PLANAR_PTYCHOGRAPHY
    """The forward model to use for physical guidance"""
    
    generator_seed: Optional[int] = None
    """The seed for the generator used by the diffusion model. This is needed to guarantee
    deterministic generation even if random seeds are used elsewhere.
    """
    
    def get_reconstructor_type(self):
        return "guided_diffusion"
    
    def check(self, options: pcapi.options.task_options.PtychographyTaskOptions) -> None:
        super().check(options)
        if self.physical_guidance_method == enums.PhysicalGuidanceMethods.RESAMPLE:
            if self.noise_scheduler != enums.NoiseSchedulers.DDIMScheduler:
                raise ValueError("ReSample is only compatible with DDIMScheduler.")


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
    
