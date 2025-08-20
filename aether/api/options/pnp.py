from dataclasses import dataclass, field
from typing import Optional, Union

import numpy as np
import ptychi.api as pcapi


@dataclass
class PriorProjectionOptions(pcapi.base.Options):
    pass


@dataclass
class ImageEditingOptions(PriorProjectionOptions):
    edit_phase: bool = True
    """If True, the phase of the object will be edited during prior projection.
    If False, the phase is kept as is if `constant_phase_value` is not provided;
    otherwise, the phase is replaced with the given constant.
    """
    
    constant_phase_value: Optional[float] = None
    """The constant phase value to use when `edit_phase` is False. If None, the phase
    is kept as is.
    """
    
    edit_magnitude: bool = False
    """If True, the magnitude of the object will be edited during prior projection.
    If False, the magnitude is kept as is if `constant_magnitude_value` is not provided;
    otherwise, the magnitude is replaced with the given constant.
    """
    
    constant_magnitude_value: Optional[float] = None
    """The constant magnitude value to use when `edit_magnitude` is False. If None, the
    magnitude is kept as is.
    """
   
    only_edit_bbox: bool = False
    """If True, only the region inside the bounding box of all probe positions will be
    edited.
    """
    
    unwrap_phase_before_editing: bool = False
    """If True, the phase of the object will be unwrapped before editing."""


@dataclass
class LEDITSPPOptions(ImageEditingOptions):
    editing_prompt: Union[str, list[str], list[list[str]]] = ""
    """The prompts of the concepts for the image editing model to add or remove. If
    given as a single string or a list of strings, the same prompt will be used for
    all slices. Multiple concepts can be given by providing a list of strings. To
    specify different prompts for different slices, provide a list of lists of strings 
    (i.e., a 2D list of strings).
    """
    
    resize_image_edited_to: Optional[tuple[int, int]] = None
    """If provided, images are resized to the given size before editing. This bypasses
    the implicit size restrctions due to the cross attention mask of LEDTS++. It helps
    the diffusion model by giving it a favorable image size such as 512x512.
    """

    text_guidance_scale: Union[float, list[float]] = 7
    """The guidance scale to use for the guided sampling. If a list of floats is provided,
    it will be assumed that each element specifies the value for a certain slice.
    """
    
    editing_threshold: Union[float, list[float]] = 0.75
    """The editing threshold. A smaller value increases the area of effect. 
    If a list of floats is provided, it will be assumed that each element specifies 
    the value for a certain slice.
    """
    
    editing_skip: Union[float, list[float]] = 0.1
    """The skip factor for the inversion process. A smaller value means more noise will
    be added to the input image, resulting in larger changes to the image.
    """
    
    remove_concept: bool = True
    """If True, the concept in the prompt will be removed from the image. Otherwise, 
    it will be added.
    """
    
    num_inference_steps: int = 50
    """The number of inference steps to use for the guided sampling."""
    
    model_path: str = "stable-diffusion-v1-5/stable-diffusion-v1-5"
    """The path to the model to use for the guided sampling."""
    
    match_stats_of_prior_projected_image: bool = False
    """If True, after each prior projection, the mean and standard deviation of the 
    prior-projected image will be matched to those of the data-projected image.
    """
    
    stats_matching_threshold: float | list[float] = 0.5
    """The absolute difference threshold used to generate the mask for stats matching.
    The algorithm calculates the absolute difference between the normalized images before
    and after editing; mean and standard deviation are only calculated on pixels where the
    absolute change is less than this threshold. Set to 0 to disable stats matching; set
    to 1 to calculate stats using all pixels. If a list of floats is provided, it will be
    assumed that each element specifies the value for a certain slice. This field is disregarded
    if `match_stats_of_prior_projected_image` is False.
    """
    
    generator_seed: Optional[int] = None
    """The seed for the generator used by the diffusion model. This is needed to guarantee
    deterministic generation even if random seeds are used elsewhere.
    """
    
    def get_reconstructor_type(self):
        return "alternating_projection"
    
    def check(self, options: pcapi.options.task_options.PtychographyTaskOptions) -> None:
        super().check(options)


@dataclass
class PnPReconstructorOptions(pcapi.base.Options):
    data_projection_options: pcapi.options.task.PtychographyTaskOptions = field(
        default_factory=pcapi.options.task.PtychographyTaskOptions
    )
    """Pty-Chi options for solving the data fidelity subproblem."""
    
    prior_projection_options: PriorProjectionOptions = field(default_factory=PriorProjectionOptions)
    """Options for solving the prior projection (image editing, artifact removal, etc.) 
    subproblem.
    """
    
    num_epochs: int = 100
    """The number of outer epochs."""
    
    num_data_projection_epochs: int = 20
    """The number of epochs for data projection."""
    
    include_dual_in_data_projection_initialization: bool = True
    """If True, the prime variable for the data projection subproblem is initialized
    as `x = v - u` instead of `x = v`.
    """

    use_prior_projected_data_as_final_result: bool = False
    """If True, the object data at the end of the reconstruction is set to the data after
    prior projection (v). Otherwise, it will be set to the data after data projection (x),
    i.e., the step before prior projection.
    """
    
    prior_projection_starting_epoch: int = 0
    """The outer iteration index at which prior projection starts."""
    
    max_num_prior_projections: int = np.inf
    """The maximum allowable number of prior projections. Prior projection no longer runs
    after exceeding this value and the reconstruction becomes purely data-driven.
    """
    
    proximal_penalty: float = 1.0
    """The penalty for the proximal term in calculating the projections. 
    ADMM reduces to alternating projection when `proximal_penalty == 0` and
    `update_relaxation == 1`.
    """
    
    update_relaxation: float = 1.0
    """The relaxation factor for the prior projection variable in ADMM. After
    each prior projection, the variable is updated as `v = gamma * v + (1 - gamma) * x`.
    ADMM reduces to alternating projection when `proximal_penalty == 0` and 
    `update_relaxation == 1`.
    """
    
    batch_size: int = 1
    """This argument currently has no effect."""
    
    displayed_loss_function: pcapi.enums.LossFunctions = None
    """This argument currently has no effect."""
    
    default_device: pcapi.enums.Devices = pcapi.enums.Devices.GPU
    """The default device to use for computation."""

    default_dtype: pcapi.enums.Dtypes = pcapi.enums.Dtypes.FLOAT32
    """The default data type to use for computation."""


@dataclass
class PnPObjectOptions(pcapi.options.base.ObjectOptions):
    
    def check(self, *args, **kwargs):
        pass
    

@dataclass
class PnPOptions(pcapi.options.task.PtychographyTaskOptions):
    
    reconstructor_options: PnPReconstructorOptions = field(default_factory=PnPReconstructorOptions)
        
    object_options: PnPObjectOptions = field(default_factory=PnPObjectOptions)


    def check(self, *args, **kwargs):
        pass
