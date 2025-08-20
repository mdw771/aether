from typing import Union, Any
import logging

import torch
import torchvision.transforms as T
from diffusers import LEditsPPPipelineStableDiffusion
from ptychi.reconstructors.base import IterativeReconstructor
from ptychi.data_structures.parameter_group import PtychographyParameterGroup
from ptychi.api.task import PtychographyTask

import aether.api as api
from aether.io import HuggingFaceModelLoader
import aether.image_proc as ip


logger = logging.getLogger(__name__)


class PnPReconstructor(IterativeReconstructor):
    
    options: api.PnPReconstructorOptions
    pipe: LEditsPPPipelineStableDiffusion

    def __init__(
        self,
        parameter_group: PtychographyParameterGroup,
        model_loader: HuggingFaceModelLoader,
        options: "api.PnPReconstructorOptions",
        *args, **kwargs
    ):
        """
        The alternating projection reconstructor. This algorithm uses a relaxed ADMM
        algorithm to integrate analytical solver and a img2img diffusion model. The
        relaxed ADMM can optionally reduce to alternating projection. 
        
        The conventions of ADMM are based on https://engineering.purdue.edu/~bouman/Plug-and-Play/webdocs/GlobalSIP2013a.pdf;
        the relaxed ADMM algorithm is based on https://arxiv.org/abs/1704.02712.
        """
        super().__init__(parameter_group, options=options, dataset=None, *args, **kwargs)
        self.check_inputs()
        
        self.model_loader = model_loader
        self.pipe: LEditsPPPipelineStableDiffusion = None
        self.forward_model = None
                
        self.x = None
        self.v = None
        self.u = None
        
        self.num_prior_projections = 0
        self.num_data_projections = 0
        
        self.image_normalizer = ip.ImageNormalizer()
        
        self.generator = None
                
    def check_inputs(self, *args, **kwargs):
        super().check_inputs(*args, **kwargs)
        
    def build(self):
        self.build_pipe()
        self.build_generator()
        super().build()
        self.build_variables()
        self.build_ptychi_task()
        self.build_counter()
        self.build_object_roi_bbox()
        
    def build_dataloader(self):
        pass

    def build_pipe(self):
        self.model_loader.load()
        self.pipe = self.model_loader.pipe
        self.pipe.to("cuda")
        
    def build_ptychi_task(self):
        self.ptychi_task = PtychographyTask(self.options.data_projection_options)
        
    def build_generator(self):
        self.generator = torch.Generator(device=self.pipe.device)
        if self.options.prior_projection_options.generator_seed is not None:
            self.generator.manual_seed(self.options.prior_projection_options.generator_seed)
        
    def build_variables(self):
        self.x = self.parameter_group.object.data.detach()
        self.v = torch.zeros_like(self.x)
        self.u = torch.zeros_like(self.x)
        
    def build_counter(self):
        super().build_counter()
        self.num_prior_projections = 0
        self.num_data_projections = 0
        
    def build_object_roi_bbox(self):
        self.parameter_group.object.update_pos_origin_coordinates(
            self.ptychi_task.reconstructor.parameter_group.probe_positions.data
        )
        self.parameter_group.object.build_roi_bounding_box(
            self.ptychi_task.reconstructor.parameter_group.probe_positions
        )
        
    def use_admm(self) -> bool:
        return (
            self.num_prior_projections < self.options.max_num_prior_projections
        ) and (
            self.current_epoch >= self.options.prior_projection_starting_epoch
        )
        
    def set_object_data_to_forward_model(self, o_hat: torch.Tensor):
        """Set object data to the object function object in the forward model. 
        We can't use object.set_data() here because it is an in-place operation
        that will break the gradient.
        """
        if isinstance(self.forward_model.object.tensor.data, torch.nn.Parameter):
            raise TypeError(
                "The tensor in the ComplexTensor of the object should not "
                "be a torch.nn.Parameter. When creating the object, set "
                "`data_as_parameter=False`."
            )
        self.forward_model.object.tensor.data = torch.stack([o_hat.real, o_hat.imag], dim=-1)
    
    def sync_data_to_object(self):
        if self.options.use_prior_projected_data_as_final_result:
            self.parameter_group.object.set_data(self.v)
        else:
            self.parameter_group.object.set_data(self.x)
            
    def get_slice_specific_option_value(
        self, 
        val: Union[Any, list[Any], list[list[Any]]],
        slice_idx: int,
        slice_value_is_list: bool = False
    ) -> Union[Any, list[Any]]:
        """Get the slice-specific value from a list of values from fields in
        the options object. If the input is a scalar, it is returned as is. 
        If it is a list, the element corresponding to `slice_idx` is returned.
        
        If `slice_value_is_list` is True, it is assumed that the value for each
        slice is a list. In that case, the input can be a list of lists.
        
        Parameters
        ----------
        val: Union[Any, list[Any], list[list[Any]]]
            The value to get the slice-specific value from.
        slice_idx: int
            The index of the slice to get the value from.
        slice_value_is_list: bool
            Whether the value for each slice is supposed to be a list.
            
        Returns
        -------
        Union[Any, list[Any]]
            The slice-specific value.
        """
        if not isinstance(val, (list, tuple)):
            if slice_value_is_list:
                return [val]
            else:
                return val
        elif isinstance(val[0], (list, tuple)):
            if slice_value_is_list:
                return val[slice_idx]
            else:
                raise ValueError(
                    "slice_value_is_list is False, but the input is a list of lists."
                )
        else:
            if slice_value_is_list:
                return val
            else:
                return val[slice_idx]
        
    def project_to_data(self):
        if self.current_epoch == 0 or not self.use_admm() or self.v.abs().max() == 0:
            x = self.x
        else:
            x = (
                self.v if not self.options.include_dual_in_data_projection_initialization 
                else self.v - self.u
            )
        x = x.detach()
        
        self.ptychi_task.reconstructor.parameter_group.object.set_data(x)
        
        for _ in range(self.options.num_data_projection_epochs):
            # Run one epoch of data fidelity term optimization with Pty-Chi.
            self.ptychi_task.run(1)
            
            # Proximal term step.
            if self.current_epoch > 0 and self.use_admm():
                x = self.ptychi_task.reconstructor.parameter_group.object.data
                x = x - self.options.proximal_penalty * (x - self.v + self.u)
                self.ptychi_task.reconstructor.parameter_group.object.set_data(x)
        self.x = self.ptychi_task.reconstructor.parameter_group.object.data
        self.num_data_projections += 1

    def project_to_prior(self):
        assert isinstance(self.options.prior_projection_options, api.LEDITSPPOptions)
        
        input = self.x + self.u

        orig_img_mag, orig_img_phase = ip.object_to_image(
            input, 
            dtype=self.pipe.unet.dtype, 
            unwrap_phase=self.options.prior_projection_options.unwrap_phase_before_editing
        )
        bbox_slicer = (slice(None),)
        if self.options.prior_projection_options.only_edit_bbox:
            bbox_slicer = self.parameter_group.object.roi_bbox.get_bbox_with_top_left_origin().get_slicer()
            orig_img_mag = orig_img_mag[:, :, *bbox_slicer]
            orig_img_phase = orig_img_phase[:, :, *bbox_slicer]
            
        orig_size = orig_img_mag.shape[-2:]
        if self.options.prior_projection_options.resize_image_edited_to is not None:
            orig_img_mag = T.Resize(self.options.prior_projection_options.resize_image_edited_to)(orig_img_mag)
            orig_img_phase = T.Resize(self.options.prior_projection_options.resize_image_edited_to)(orig_img_phase)

        edited_phase_imgs = []
        edited_mag_imgs = []
        for i_part, orig_img in enumerate([orig_img_phase, orig_img_mag]):
            if i_part == 0 and not self.options.prior_projection_options.edit_phase:
                if self.options.prior_projection_options.constant_phase_value is None:
                    edited_phase_imgs.append(orig_img)
                else:
                    edited_phase_imgs.append(torch.full_like(orig_img, self.options.prior_projection_options.constant_phase_value))
            elif i_part == 1 and not self.options.prior_projection_options.edit_magnitude:
                if self.options.prior_projection_options.constant_magnitude_value is None:
                    edited_mag_imgs.append(orig_img)
                else:
                    edited_mag_imgs.append(torch.full_like(orig_img, self.options.prior_projection_options.constant_magnitude_value))
            else:
                for i_slice, orig_img_slice in enumerate(orig_img):
                    orig_img_slice_normalized = self.image_normalizer.normalize(orig_img_slice.float()).to(self.pipe.unet.dtype)
                    
                    _ = self.pipe.invert(
                        image=orig_img_slice_normalized,
                        num_inversion_steps=self.options.prior_projection_options.num_inference_steps,
                        skip=self.get_slice_specific_option_value(
                            self.options.prior_projection_options.editing_skip, i_slice
                        ),
                        generator=self.generator
                    )

                    img_slice = self.pipe(
                        editing_prompt=self.get_slice_specific_option_value(
                            self.options.prior_projection_options.editing_prompt, i_slice, slice_value_is_list=True
                        ),
                        reverse_editing_direction=self.options.prior_projection_options.remove_concept,
                        edit_guidance_scale=self.get_slice_specific_option_value(
                            self.options.prior_projection_options.text_guidance_scale, i_slice
                        ),
                        edit_threshold=self.get_slice_specific_option_value(
                            self.options.prior_projection_options.editing_threshold, i_slice
                        ),
                        generator=self.generator
                    )
                    
                    img_slice = ip.pil_image_to_tensor(img_slice.images[0], dtype=self.x.real.dtype, device=self.x.device)
                    
                    # Match mean and std within ROI.
                    if self.options.prior_projection_options.match_stats_of_prior_projected_image:
                        if self.options.prior_projection_options.only_edit_bbox:
                            roi_bbox = (slice(None), slice(None))
                        else:
                            roi_bbox = self.parameter_group.object.roi_bbox.get_bbox_with_top_left_origin().get_slicer()
                        threshold = self.get_slice_specific_option_value(
                            self.options.prior_projection_options.stats_matching_threshold, i_slice, slice_value_is_list=False
                        )
                        if threshold > 0:
                            mask = (img_slice - orig_img_slice_normalized[None, ...]).abs() < threshold
                            mask_selector = torch.zeros_like(mask, dtype=torch.bool)
                            mask_selector[(0, 0, *roi_bbox)] = True
                            mask = mask & mask_selector
                            if torch.count_nonzero(mask) > 0:
                                img_slice = ip.match_mean_std(img_slice, orig_img_slice_normalized[None, ...], mask=mask)
                            else:
                                logger.info("Skipping stats matching because no hot pixels are found within ROI.")
                            
                    img_slice = self.image_normalizer.unnormalize(img_slice)
                    if i_part == 0:
                        edited_phase_imgs.append(img_slice)
                    else:
                        edited_mag_imgs.append(img_slice)
        edited_phase_imgs = torch.cat(edited_phase_imgs, dim=0)
        edited_mag_imgs = torch.cat(edited_mag_imgs, dim=0)
        
        if self.options.prior_projection_options.resize_image_edited_to is not None:
            edited_mag_imgs = T.Resize(orig_size)(edited_mag_imgs)
            edited_phase_imgs = T.Resize(orig_size)(edited_phase_imgs)
        
        edited_obj = ip.image_to_object(
            img_mag=edited_mag_imgs,
            img_phase=edited_phase_imgs
        )
        if self.options.prior_projection_options.only_edit_bbox:
            input[:, *bbox_slicer] = edited_obj
            v = input
        else:
            v = edited_obj
        self.v = v
        self.num_prior_projections += 1
        
    def relax_prior_projection_variable(self):
        self.v = self.options.update_relaxation * self.v + (1 - self.options.update_relaxation) * self.x

    def update_dual(self):
        self.u = self.u + self.x - self.v
        
    def run_admm_epoch(self):
        self.project_to_data()
        if self.use_admm():
            self.project_to_prior()
            self.relax_prior_projection_variable()
            self.update_dual()
            
    def run_pre_epoch_hooks(self):
        pass
    
    def run(self, n_epochs: int = None):
        n_epochs = self.options.num_epochs if n_epochs is None else n_epochs
        with torch.no_grad():
            for _ in range(n_epochs):
                self.run_pre_epoch_hooks()
                self.run_admm_epoch()
                self.sync_data_to_object()
                
                self.pbar.update(1)
                self.current_epoch += 1
