from typing import Literal, Optional
import logging

import numpy as np
import torch
from diffusers import StableDiffusionXLPipeline
from ptychi.reconstructors.ad_ptychography import AutodiffPtychographyReconstructor
from ptychi.data_structures.parameter_group import PtychographyParameterGroup
import ptychi.maps as pcmaps
from ptychi.io_handles import PtychographyDataset

import firefly.maths as maths
import firefly.api as api
from firefly.io import HuggingFaceStableDiffusionModelLoader
import firefly.image_proc as ip


logger = logging.getLogger(__name__)


class AlternatingProjectionReconstructor(AutodiffPtychographyReconstructor):

    def __init__(
        self,
        parameter_group: PtychographyParameterGroup,
        dataset: PtychographyDataset,
        model_loader: HuggingFaceStableDiffusionModelLoader,
        options: "api.GuidedDiffusionReconstructorOptions",
        *args, **kwargs
    ):
        super().__init__(parameter_group, dataset=dataset, options=options, *args, **kwargs)
        self.check_inputs()
        
        self.model_loader = model_loader
        self.pipe: StableDiffusionXLPipeline = None
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
        
    def image_to_object(
        self, 
        img: torch.Tensor, 
        representation: Literal["real_imag", "mag_phase"] = "mag_phase"
    ) -> torch.Tensor:
        """Convert a pixel-space image to a complex object tensor.
        
        Parameters
        ----------
        img: torch.Tensor
            A (1, 3, h, w) tensor giving the decoded image.
            
        Returns
        -------
        torch.Tensor
            A (1, h, w) tensor giving the complex object.
        """
        if img.shape[0] != 1:
            raise ValueError("The length of the batch dimension of img must be 1.")
        
        if representation == "real_imag":
            real = img[:, 0, ...] + img[:, 2, ...] * 0.5
            imag = img[:, 1, ...] + img[:, 2, ...] * 0.5
            obj = real + 1j * imag
        elif representation == "mag_phase":
            mag = 1
            phase = img.mean(1)
            obj = mag * torch.exp(1j * phase)
        else:
            raise ValueError(f"Invalid representation: {representation}")
        obj = maths.TypeCastFunction.apply(obj, torch.complex64)
        return obj
    
    def object_to_image(
        self, 
        obj: torch.Tensor, 
        representation: Literal["real_imag", "mag_phase", "single_channel"] = "mag_phase"
    ) -> torch.Tensor:
        """Convert a complex object tensor to an image assumed by the encoder.
        
        Parameters
        ----------
        obj: torch.Tensor
            A (1, h, w) tensor giving the complex object.
            
        Returns
        -------
        torch.Tensor
            A (1, 3, h, w) tensor giving the image.
        """
        if representation == "real_imag":
            img = torch.stack([
                obj.real,
                obj.imag,
                0.5 * obj.real + 0.5 * obj.imag
            ], dim=1)
        elif representation == "mag_phase":
            phase = obj.angle()
            # phase = self.preconditioned_phase_unwrap(obj)
            img = phase.unsqueeze(1)
            img = img.repeat(1, 3, 1, 1)
        elif representation == "single_channel":
            if obj.dtype.is_complex:
                raise ValueError("Single channel representation does not support complex numbers.")
            img = obj.unsqueeze(1)
            img = img.repeat(1, 3, 1, 1)
        else:
            raise ValueError(f"Invalid representation: {representation}")
        img = img.to(self.pipe.dtype)
        return img
        
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
        
    def project_to_data(self):
        x_tilde = self.v - self.u
        
        self.x = self.x.requires_grad_(True)
        optimizer = torch.optim.Adam([self.x], lr=1e-3)
        
        with torch.enable_grad():            
            for i_data_proj_epoch in range(self.options.num_data_projection_epochs):
                for batch_data in self.dataloader:
                    self.x.grad = None
                    self.set_object_data_to_forward_model(self.x)
                    
                    input_data, y_true = self.prepare_batch_data(batch_data)
                    y_pred = self.forward_model(*input_data)
                    
                    # Data fidelity term.
                    batch_loss = self.loss_function(
                        y_pred[:, self.dataset.valid_pixel_mask], y_true[:, self.dataset.valid_pixel_mask]
                    )
                                        
                    # Proximal term.
                    reg_prox = self.options.proximal_penalty / 2 * (self.x - x_tilde).norm() ** 2
                    batch_loss += reg_prox

                    batch_loss.backward(retain_graph=True)
                    self.run_post_differentiation_hooks(input_data, y_true)
                    optimizer.step()
                    self.step_all_optimizers()
                    self.forward_model.zero_grad()
                    
                    self.loss_tracker.update_batch_loss_with_value(batch_loss.item())
                self.loss_tracker.conclude_epoch()
                self.loss_tracker.print_latest()
        self.x = self.x.detach()
    
    def project_to_prior(self):
        img = self.object_to_image(self.x)
        img = self.image_normalizer.normalize(img)
        img = self.pipe(
            prompt=self.options.prompt,
            image=img,
            strength=self.options.prior_strength,
            num_inference_steps=self.options.num_inference_steps,
            guidance_scale=self.options.text_guidance_scale,
            negative_prompt=self.options.negative_prompt,
        )
        img = torch.tensor(np.array(img.images[0]), dtype=self.x.real.dtype, device=self.x.device)
        img = img / 255.0
        img = img.permute(2, 0, 1)[None, ...]
        img = self.image_normalizer.unnormalize(img)
        self.x = self.image_to_object(img)
    
    def update_dual(self):
        # self.u = self.u + self.x - self.v
        pass
        
    def run_admm_epoch(self):
        self.project_to_data()
        self.project_to_prior()
        self.update_dual()
    
    def run(self, n_epochs: int = None):
        n_epochs = self.options.num_epochs if n_epochs is None else n_epochs
        with torch.no_grad():
            for _ in range(n_epochs):
                # self.run_pre_epoch_hooks()
                self.run_admm_epoch()
                self.pbar.update(1)