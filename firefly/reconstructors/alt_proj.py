from typing import Literal, Optional
import logging

import numpy as np
import torch
from diffusers import LEditsPPPipelineStableDiffusion
from ptychi.reconstructors.ad_ptychography import AutodiffPtychographyReconstructor
from ptychi.data_structures.parameter_group import PtychographyParameterGroup
import ptychi.maps as pcmaps
from ptychi.io_handles import PtychographyDataset

import firefly.maths as maths
import firefly.api as api
from firefly.io import HuggingFaceModelLoader
import firefly.image_proc as ip


logger = logging.getLogger(__name__)


class AlternatingProjectionReconstructor(AutodiffPtychographyReconstructor):
    
    options: api.AlternatingProjectionReconstructorOptions
    pipe: LEditsPPPipelineStableDiffusion

    def __init__(
        self,
        parameter_group: PtychographyParameterGroup,
        dataset: PtychographyDataset,
        model_loader: HuggingFaceModelLoader,
        options: "api.GuidedDiffusionReconstructorOptions",
        *args, **kwargs
    ):
        """
        The alternating projection reconstructor. This algorithm uses a relaxed ADMM
        algorithm to integrate analytical solver and a img2img diffusion model. The
        relaxed ADMM can optionally reduce to alternating projection. 
        
        The conventions of ADMM are based on https://engineering.purdue.edu/~bouman/Plug-and-Play/webdocs/GlobalSIP2013a.pdf;
        the relaxed ADMM algorithm is based on https://arxiv.org/abs/1704.02712.
        """
        super().__init__(parameter_group, dataset=dataset, options=options, *args, **kwargs)
        self.check_inputs()
        
        self.model_loader = model_loader
        self.pipe: LEditsPPPipelineStableDiffusion = None
        self.loss_function = pcmaps.get_loss_function_by_enum(options.loss_function)()
        self.forward_model = None
                
        self.x = None
        self.v = None
        self.u = None
        
        self.image_normalizer = ip.ImageNormalizer()
        
    def build(self):
        self.build_pipe()
        super().build()
        self.build_variables()
    def build_pipe(self):
        self.model_loader.load()
        self.pipe = self.model_loader.pipe
        self.pipe.to("cuda")
        
    def build_forward_model(self):
        self.forward_model_params["wavelength_m"] = self.dataset.wavelength_m
        self.forward_model_params["detector_size"] = tuple(self.dataset.patterns.shape[-2:])
        self.forward_model_params["free_space_propagation_distance_m"] = self.dataset.free_space_propagation_distance_m
        self.forward_model_params["pad_for_shift"] = self.options.forward_model_options.pad_for_shift
        self.forward_model_params["low_memory_mode"] = self.options.forward_model_options.low_memory_mode
        self.forward_model = self.forward_model_class(
            self.parameter_group, **self.forward_model_params
        )
        
    def build_variables(self):
        self.x = self.parameter_group.object.data.detach()
        self.v = torch.zeros_like(self.x)
        self.u = torch.zeros_like(self.x)
        
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
        
    def project_to_data(self):
        if self.current_epoch == 0:
            x = self.x
        else:
            x = self.v - self.u
        x = x.requires_grad_(True)
        optimizer = torch.optim.Adam([x], lr=1e-3)
        
        with torch.enable_grad():            
            for i_data_proj_epoch in range(self.options.num_data_projection_epochs):
                for batch_data in self.dataloader:
                    x.grad = None
                    self.set_object_data_to_forward_model(x)
                    
                    input_data, y_true = self.prepare_batch_data(batch_data)
                    y_pred = self.forward_model(*input_data)
                    
                    # Data fidelity term.
                    batch_loss = self.loss_function(
                        y_pred[:, self.dataset.valid_pixel_mask], y_true[:, self.dataset.valid_pixel_mask]
                    )
                                        
                    # Proximal term (skipped for the first outer epoch).
                    if self.options.proximal_penalty > 0 and self.current_epoch > 0: 
                        reg_prox = self.options.proximal_penalty / 2 * (
                            x - self.v + self.u
                        ).norm() ** 2
                        batch_loss += reg_prox

                    batch_loss.backward(retain_graph=True)
                    self.run_post_differentiation_hooks(input_data, y_true)
                    optimizer.step()
                    self.step_all_optimizers()
                    self.forward_model.zero_grad()
                    
                    self.loss_tracker.update_batch_loss_with_value(batch_loss.item())
                self.loss_tracker.conclude_epoch()
                self.loss_tracker.print_latest()
        self.x = x.detach()
    
    def project_to_prior(self):
        input = self.x + self.u
        _, orig_img = ip.object_to_image(input, dtype=self.pipe.unet.dtype)
        orig_img = self.image_normalizer.normalize(orig_img.float()).to(self.pipe.unet.dtype)
        
        _ = self.pipe.invert(
            image=orig_img,
            num_inversion_steps=50,
            skip=0.1
        )
        
        if isinstance(self.options.editing_prompt, str):
            editing_prompt = [self.options.editing_prompt]
        else:
            editing_prompt = self.options.editing_prompt
        img = self.pipe(
            editing_prompt=editing_prompt,
            reverse_editing_direction=self.options.remove_concept,
            edit_guidance_scale=self.options.text_guidance_scale,
            edit_threshold=0.75,
        )
        
        img = ip.pil_image_to_tensor(img.images[0], dtype=self.x.real.dtype, device=self.x.device)
        img = self.image_normalizer.unnormalize(img)
        
        # Match mean and std within ROI.
        if self.options.matched_stats_of_prior_projected_image:
            bbox = self.parameter_group.object.roi_bbox.get_bbox_with_top_left_origin().get_slicer()
            img = ip.match_mean_std(img, orig_img, (0, 0, *bbox))
        
        v = ip.image_to_object(img_mag=self.x.abs().unsqueeze(1).repeat(1, 3, 1, 1), img_phase=img)
        self.v = self.options.update_relaxation * v + (1 - self.options.update_relaxation) * self.x
    
    def update_dual(self):
        self.u = self.u + self.x - self.v
        
    def run_admm_epoch(self):
        self.project_to_data()
        self.project_to_prior()
        self.update_dual()
    
    def run(self, n_epochs: int = None):
        n_epochs = self.options.num_epochs if n_epochs is None else n_epochs
        with torch.no_grad():
            for _ in range(n_epochs):
                self.run_admm_epoch()
                self.sync_data_to_object()
                
                self.pbar.update(1)
                self.current_epoch += 1
