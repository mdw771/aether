from typing import Literal, Optional
import logging

import numpy as np
from PIL import Image
import torch
from diffusers import StableDiffusion3Pipeline, StableDiffusionXLPipeline
from ptychi.reconstructors.ad_ptychography import AutodiffPtychographyReconstructor
from ptychi.data_structures.parameter_group import PtychographyParameterGroup
import ptychi.maps as pcmaps
from ptychi.io_handles import PtychographyDataset
import ptychi.image_proc as ip

import firefly.maths as maths
import firefly.api as api
from firefly.io import HuggingFaceStableDiffusionModelLoader
import firefly.maps as maps
import firefly.util as util


logger = logging.getLogger(__name__)


class GuidedDiffusionReconstructor(AutodiffPtychographyReconstructor):
    
    
    class TextEmbeddings:
        prompt_embeds: torch.Tensor
        negative_prompt_embeds: torch.Tensor
        pooled_prompt_embeds: torch.Tensor
        negative_pooled_prompt_embeds: torch.Tensor
        add_time_ids: torch.Tensor
        negative_add_time_ids: torch.Tensor
        add_text_embeds: torch.Tensor
    
    options: "api.GuidedDiffusionReconstructorOptions"
    
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
        
        self.current_denoise_step = 0
        self.do_classifier_free_guidance = self.options.text_guidance_scale > 1 
        self.text_embeddings = self.TextEmbeddings()
        
        self.sampled_image: torch.Tensor = None
        self.sampled_image_pil: Image.Image = None
                
    def check_inputs(self):
        super().check_inputs()
        if self.parameter_group.object.optimizable:
            raise ValueError("The object must not be optimizable for physics-guided sampling.")
        
    def build(self):
        self.build_pipe()
        super().build()
        
    def build_pipe(self):
        # Load pipe to CPU.
        self.model_loader.load()
        self.pipe = self.model_loader.pipe
        
        # Change scheduler.
        with util.ignore_default_device():
            self.pipe.scheduler = maps.get_noise_scheduler(
                self.options.noise_scheduler
            ).from_config(self.pipe.scheduler.config)
        
        # Set timesteps.
        self.pipe.scheduler.set_timesteps(
            self.options.num_inference_steps, device=torch.device("cpu")
        )
        
        # VAE requires float32 to prevent overflow. 
        # `self.decode_latent` and `self.encode_image` contain upcasting, but they
        # run into issues during backprop when the decoded image is casted to float32
        # prior to the physics forward model. Thus, we avoid the upcasting by converting
        # the VAE to float32 beforehand, and casting the latent to float32 before the
        # automatic differentiation graph of physical guidance.
        self.pipe.vae = self.pipe.vae.to(torch.float32)
        
        # Move pipe to GPU.
        self.pipe = self.pipe.to("cuda")
        
    def build_counter(self):
        super().build_counter()
        self.pbar.total = self.options.num_inference_steps
        
    def build_forward_model(self):
        self.forward_model_params["wavelength_m"] = self.dataset.wavelength_m
        self.forward_model_params["detector_size"] = tuple(self.dataset.patterns.shape[-2:])
        self.forward_model_params["free_space_propagation_distance_m"] = self.dataset.free_space_propagation_distance_m
        self.forward_model_params["pad_for_shift"] = self.options.forward_model_options.pad_for_shift
        self.forward_model_params["low_memory_mode"] = self.options.forward_model_options.low_memory_mode
        self.forward_model = self.forward_model_class(
            self.parameter_group, **self.forward_model_params
        )
        
    def get_denoising_step_from_timestep(self, t: int):
        """Get the denoising step from the timestep.
        
        Parameters
        ----------
        t: int
            The timestep in `self.pipe.scheduler.timesteps`.
            
        Returns
        -------
        int
            The timestep in `self.pipe.scheduler.timesteps`.
        """
        return torch.where(self.pipe.scheduler.timesteps == t)[0][0]
            
    def get_alpha_prod_t(self, *, t: Optional[int] = None, denoising_step: Optional[int] = None):
        """Get the cumulative alpha product at a specific timestep or
        denoising step.
        
        Parameters
        ----------
        t: int
            The timestep in `self.pipe.scheduler.timesteps`.
        denoising_step: int
            The index of the denoising step, starting from 0 during
            inference.
            
        Returns
        -------
        float
            The cumulative alpha product at the given timestep or denoising step.
        """
        if t is not None and denoising_step is not None:
            raise ValueError("Either t or denoising_step must be provided, not both.")
        if t is None and denoising_step is None:
            raise ValueError("Either t or denoising_step must be provided.")
        if denoising_step is not None:
            t = self.pipe.scheduler.timesteps[denoising_step]
        ind = int(t)
        return self.pipe.scheduler.alphas_cumprod[ind]
        
    def preconditioned_phase_unwrap(self, obj: torch.Tensor) -> torch.Tensor:
        """Phase unwrapping with preconditioning. The constant phase offset
        is removed from the unwrapped phase using the phase of the center pixel
        of the input.
        
        Parameters
        ----------
        obj: torch.Tensor
            A (n_slices, h, w) complex tensor giving the object to be unwrapped.
            
        Returns
        -------
        torch.Tensor
            A (n_slices, h, w) tensor giving the unwrapped phase.
        """
        phase = []
        for obj_slice in obj:
            phase_center = obj_slice[obj_slice.shape[0] // 2, obj_slice.shape[1] // 2].angle()
            phase_slice = ip.unwrap_phase_2d(
                obj_slice,
                image_grad_method="fourier_differentiation",
                image_integration_method="fourier"
            )
            phase_center_new = phase_slice[obj_slice.shape[0] // 2, obj_slice.shape[1] // 2]
            phase_slice = phase_slice - phase_center_new + phase_center
            phase.append(phase_slice)
        phase = torch.stack(phase)
        phase = self.parameter_group.object.preconditioner * phase
        return phase
        
    def prepare_initial_latent(self) -> torch.Tensor:
        """Get the initial latent code. The size of the latent is determined
        by the pixel-space image size and the downscaling factor of the image
        autoencoder. In general, the size is given by
        
        ```2 ** (len(self.vae.config.block_out_channels) - 1)``` 
        
        and is 8 for Stable Diffusion 3.5. The number of channels is given
        by the number of input channels of the U-net, as in
        
        ```self.pipe.unet.config.in_channels```
        
        Returns
        -------
        torch.Tensor
            The initial latent code of shape (1, in_channels, h, w).
        """
        z = self.pipe.prepare_latents(
            batch_size=1,
            num_channels_latents=self.pipe.unet.config.in_channels,
            height=self.parameter_group.object.lateral_shape[0],
            width=self.parameter_group.object.lateral_shape[1],
            dtype=self.text_embeddings.prompt_embeds.dtype,
            device=self.text_embeddings.prompt_embeds.device,
            generator=None,
            latents=None
        )
        return z
        
    def noise_step(self, z_t: torch.Tensor, t: float, by: int):
        """Add noise to the latent code at timestep t for given timesteps.
        
        With the noisy image at timestep t given by
        ```
        x_t = \sqrt{alpha_prod_t} x_0 + \sqrt{1 - alpha_prod_t} \epsilon
        ```
        One can find the relation between two images at different timesteps,
        t1 and t2, by eliminating x_0. 
        
        Parameters
        ----------
        z_t: torch.Tensor
            The latent code to be denoised.
        t: float
            The current timestep.
        by: int
            The number of inference steps to add noise by.
            
        Returns
        -------
        torch.Tensor
            The latent code after adding noise.
        """
        ind_t = self.get_denoising_step_from_timestep(t)
        
        # `self.pipe.scheduler.timesteps` is in reverse order (front = noisier).
        if ind_t - by < 0:
            raise ValueError("The number of steps to add noise is out of bounds.")
        
        # `alphas_cumprod` has a length of 1000, not the number of inference steps.
        alpha_prod_t1 = self.get_alpha_prod_t(denoising_step=ind_t)
        alpha_prod_t2 = self.get_alpha_prod_t(denoising_step=ind_t - by)
        
        # Sample random noise
        noise = torch.randn_like(z_t)
        
        # Calculate the noisy sample x_{t+n} from z_t using the noise schedule
        z_t_plus_n = torch.sqrt(alpha_prod_t2 / alpha_prod_t1) * z_t + \
                     torch.sqrt(1 - alpha_prod_t2 / alpha_prod_t1) * noise
        
        return z_t_plus_n
    
    def physical_guidance_step(self, z_t: torch.Tensor, z_0_hat: torch.Tensor):
        """Update the latent code using the physical guidance.
        
        Parameters
        ----------
        z_t: torch.Tensor
            The latent code at time step t.
        z_0_hat: torch.Tensor
            The estimated noise-free image at time step 0.
        """
        z_0_hat = z_0_hat.to(torch.float32)
        z_t = z_t.to(torch.float32)
        if self.options.physical_guidance_method == api.enums.PhysicalGuidanceMethods.SCORE:
            score = self.calculate_physical_guidance_score(z_0_hat)
            score = self.remove_score_function_outliers(score)
            z_t = z_t - self.options.physical_guidance_scale * score
        elif self.options.physical_guidance_method == api.enums.PhysicalGuidanceMethods.RESAMPLE:
            z_0_hat = self.calculate_physical_guidance_optimal_z(z_0_hat)
            z_t = self.stochastic_resample(self.current_denoise_step + 1, z_0_hat, z_t)
        else:
            raise ValueError(f"Invalid physical guidance method: {self.options.physical_guidance_method}")
        return z_t.to(self.pipe.unet.dtype)
        
    def decode_latent(self, z: torch.Tensor):
        """Decode the latent code to the image space and convert it to
        the structure compatible with the forward model.
        
        Parameters
        ----------
        z: torch.Tensor
            The latent code to be decoded.
            
        Returns
        -------
        torch.Tensor
            The decoded image in the structure compatible with the forward model.
        """
        # make sure the VAE is in float32 mode, as it overflows in float16
        needs_upcasting = self.pipe.vae.dtype == torch.float16 and self.pipe.vae.config.force_upcast

        if needs_upcasting:
            self.pipe.upcast_vae()
            z = z.to(next(iter(self.pipe.vae.post_quant_conv.parameters())).dtype)

        # unscale/denormalize the latents
        # denormalize with the mean and std if available and not None
        has_latents_mean = hasattr(self.pipe.vae.config, "latents_mean") and self.pipe.vae.config.latents_mean is not None
        has_latents_std = hasattr(self.pipe.vae.config, "latents_std") and self.pipe.vae.config.latents_std is not None
        if has_latents_mean and has_latents_std:
            latents_mean = (
                torch.tensor(self.pipe.vae.config.latents_mean).view(1, 4, 1, 1).to(z.device, z.dtype)
            )
            latents_std = (
                torch.tensor(self.pipe.vae.config.latents_std).view(1, 4, 1, 1).to(z.device, z.dtype)
            )
            z = z * latents_std / self.pipe.vae.config.scaling_factor + latents_mean
        else:
            z = z / self.pipe.vae.config.scaling_factor

        image = self.pipe.vae.decode(z, return_dict=False)[0]

        # cast back to fp16 if needed
        if needs_upcasting:
            self.pipe.vae = maths.TypeCastFunction.apply(self.pipe.vae, self.pipe.dtype)
            image = maths.TypeCastFunction.apply(image, self.pipe.dtype)
        return image
    
    def encode_image(self, img: torch.Tensor):
        """Encode the image to the latent space.
        
        Parameters
        ----------
        img: torch.Tensor
            The image to be encoded.
            
        Returns
        -------
        torch.Tensor
            The encoded latent code.
        """
        # make sure the VAE is in float32 mode, as it overflows in float16
        needs_upcasting = self.pipe.vae.dtype == torch.float16 and self.pipe.vae.config.force_upcast

        if needs_upcasting:
            self.pipe.upcast_vae()
            img = img.to(self.pipe.vae.dtype)
            
        z = self.pipe.vae.encode(img).latent_dist.mode()

        # unscale/denormalize the latents
        # denormalize with the mean and std if available and not None
        has_latents_mean = hasattr(self.pipe.vae.config, "latents_mean") and self.pipe.vae.config.latents_mean is not None
        has_latents_std = hasattr(self.pipe.vae.config, "latents_std") and self.pipe.vae.config.latents_std is not None
        if has_latents_mean and has_latents_std:
            latents_mean = (
                torch.tensor(self.pipe.vae.config.latents_mean).view(1, 4, 1, 1).to(z.device, z.dtype)
            )
            latents_std = (
                torch.tensor(self.pipe.vae.config.latents_std).view(1, 4, 1, 1).to(z.device, z.dtype)
            )
            z = (z - latents_mean) / latents_std * self.pipe.vae.config.scaling_factor
        else:
            z = z * self.pipe.vae.config.scaling_factor

        # cast back to fp16 if needed
        if needs_upcasting:
            self.pipe.vae.to(self.pipe.dtype)
            z = z.to(self.pipe.dtype)
        return z
    
    def prepare_added_time_ids_and_embeddings(self):
        """Prepare the added time ids and embeddings.
        """
        add_text_embeds = self.text_embeddings.pooled_prompt_embeds
        if self.pipe.text_encoder_2 is None:
            text_encoder_projection_dim = int(self.text_embeddings.pooled_prompt_embeds.shape[-1])
        else:
            text_encoder_projection_dim = self.pipe.text_encoder_2.config.projection_dim

        add_time_ids = self.pipe._get_add_time_ids(
            tuple(self.parameter_group.object.lateral_shape),
            crops_coords_top_left=(0, 0),
            target_size=tuple(self.parameter_group.object.lateral_shape),
            dtype=self.text_embeddings.prompt_embeds.dtype,
            text_encoder_projection_dim=text_encoder_projection_dim,
        )
        negative_add_time_ids = add_time_ids

        if self.do_classifier_free_guidance:
            self.text_embeddings.prompt_embeds = torch.cat(
                [self.text_embeddings.negative_prompt_embeds, self.text_embeddings.prompt_embeds], 
                dim=0
            )
            add_text_embeds = torch.cat(
                [self.text_embeddings.negative_pooled_prompt_embeds, add_text_embeds], 
                dim=0
            )
            add_time_ids = torch.cat([negative_add_time_ids, add_time_ids], dim=0)
            
        add_time_ids = add_time_ids.repeat(1, 1)
        
        self.text_embeddings.add_time_ids = add_time_ids
        self.text_embeddings.negative_add_time_ids = negative_add_time_ids
        self.text_embeddings.add_text_embeds = add_text_embeds
    
    def denoise_step(self, z_t: torch.Tensor, t: int, step_index: Optional[int] = None):
        """Denoise the latent code by one step. Text conditioning is added
        to the noise through classifier-free guidance.
        The input is detached, so this function is not differentiable.
        
        Parameters
        ----------
        z_t: torch.Tensor
            The latent code to be denoised.
        t: int
            The timestep to denoise at.
        step_index: int
            The index of the current denoising step. It will be used to override
            the `step_index` attribute of the scheduler.
            
        Returns
        -------
        z_tm1: torch.Tensor
            The denoised latent code (x_{t-1}).
        z_0_hat: torch.Tensor
            The estimated noise-free image at time step 0.
        """
        latent_model_input = torch.cat([z_t] * 2) if self.do_classifier_free_guidance else z_t
        latent_model_input = self.pipe.scheduler.scale_model_input(latent_model_input, t)
        
        added_cond_kwargs = {"text_embeds": self.text_embeddings.add_text_embeds, "time_ids": self.text_embeddings.add_time_ids}
                    
        # Standard denoising step.
        noise_pred = self.pipe.unet(
            latent_model_input,
            t,
            encoder_hidden_states=self.text_embeddings.prompt_embeds,
            added_cond_kwargs=added_cond_kwargs,
            return_dict=False
        )[0]
        
        # Classifier-free (text) guidance step.
        if self.do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + self.options.text_guidance_scale * (noise_pred_text - noise_pred_uncond)
        
        if step_index is not None:
            self.pipe.scheduler._step_index = step_index
        
        step_output = self.pipe.scheduler.step(noise_pred, t, z_t, return_dict=True)
        z_tm1 = step_output.prev_sample
        z_0_hat = step_output.pred_original_sample
        
        if z_tm1.dtype != z_t.dtype:
            z_tm1 = z_tm1.to(z_t.dtype)
        if z_0_hat.dtype != z_t.dtype:
            z_0_hat = z_0_hat.to(z_t.dtype)
        return z_tm1, z_0_hat
    
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
        
    def initialize_gradients(self, z: torch.Tensor):
        z.grad = None
        
    def calculate_physical_guidance_score(self, z: torch.Tensor):
        """Denoise the latent code by one step, decode it, and calculate
        the score function of the physical model $f$, given by
        $\nabla_{z_{t-1}} || f(D(z_{t-1})) - y ||^2$,  where $D$ is the
        decoder and $f$ is the forward model.
        
        Parameters
        ----------
        z: torch.Tensor
            A (1, in_channels, h, w) tensor giving the latent code.
            
        Returns
        -------
        torch.Tensor
            The score function of the physical model.
        """
        if z.shape[0] != 1:
            raise ValueError("The length of the batch dimension of z_t must be 1.")
        
        accumulated_grad_phase = torch.zeros_like(z)
        with torch.enable_grad():
            z = z.requires_grad_(True)
            
            decoded_image = self.decode_latent(z)
            o_hat = self.image_to_object(decoded_image)
            self.set_object_data_to_forward_model(o_hat)
            
            for batch_data in self.dataloader:
                self.initialize_gradients(z)
                input_data, y_true = self.prepare_batch_data(batch_data)
                y_pred = self.forward_model(*input_data)
                batch_loss = self.loss_function(
                    y_pred[:, self.dataset.valid_pixel_mask], y_true[:, self.dataset.valid_pixel_mask]
                )
                batch_loss.backward(retain_graph=True)
                self.step_all_optimizers()
                accumulated_grad_phase += z.grad * y_pred.numel()
                self.forward_model.zero_grad()
        
        accumulated_grad_phase = accumulated_grad_phase / self.dataset.patterns.numel()
        return accumulated_grad_phase
    
    def remove_score_function_outliers(self, score: torch.Tensor):
        """Remove outliers from the score function.
        
        Parameters
        ----------
        score: torch.Tensor
            A (1, c, h, w) tensor giving the score function.
            
        Returns
        -------
        torch.Tensor
            The processed score function.
        """
        for c in range(score.shape[1]):
            abs_score_channel = score[:, c, ...].abs()
            q = torch.quantile(abs_score_channel, 0.99)
            score[:, c, ...] = score[:, c, ...] * (abs_score_channel < q)
        return score
    
    def calculate_physical_guidance_optimal_z(self, z: torch.Tensor):
        """Compute the optimal latent code that minimizes the physical loss.
        
        Parameters
        ----------
        z: torch.Tensor
            The estimated latent at time step 0.
            
        Returns
        -------
        torch.Tensor
            The resampled latent code.
        """
        if z.shape[0] != 1:
            raise ValueError("The length of the batch dimension of z_t must be 1.")
        
        z_optimizer = pcmaps.get_optimizer_by_enum(
            self.options.resample_options.optimizer
        )(
            [z], lr=self.options.resample_options.step_size
        )
        
        with torch.enable_grad():
            z = z.requires_grad_(True)
            for i in range(self.options.resample_options.num_z_optimization_epochs):
                epoch_loss = 0
                for batch_data in self.dataloader:
                    decoded_image = self.decode_latent(z)
                    o_hat = self.image_to_object(decoded_image)
                    self.set_object_data_to_forward_model(o_hat)
                    
                    input_data, y_true = self.prepare_batch_data(batch_data)
                    y_pred = self.forward_model(*input_data)
                    batch_loss = self.loss_function(
                        y_pred[:, self.dataset.valid_pixel_mask], y_true[:, self.dataset.valid_pixel_mask]
                    )
                    batch_loss.backward(retain_graph=True)
                    z_optimizer.step()
                    self.step_all_optimizers()
                    self.forward_model.zero_grad()
                    epoch_loss += batch_loss.item()
                logger.info(f"z-optimization epoch {i} loss: {epoch_loss / len(self.dataloader)}")
        
        z = z.detach()
        return z
    
    def stochastic_resample(self, denoising_step: int, z_0_hat: torch.Tensor, z_t_prime: torch.Tensor):
        """Stochastically resample the latent code.
        
        Parameters
        ----------
        denoising_step: int
            The index of the current denoising step.
        z_0_hat: torch.Tensor
            The estimated latent at time step 0.
        z_t_prime: torch.Tensor
            The unconditionally denoised latent at time step t.
            
        Returns
        -------
        torch.Tensor
            The resampled latent code.
        """
        sigma_t_sq = self.get_sigma_t_sq(denoising_step)
        alpha_prod_t = self.get_alpha_prod_t(denoising_step=denoising_step)
        z = sigma_t_sq * np.sqrt(alpha_prod_t) * z_0_hat + (1 - alpha_prod_t) * z_t_prime
        z = z / (sigma_t_sq + (1 - alpha_prod_t))
        if denoising_step < len(self.pipe.scheduler.timesteps) - 1:
            n = torch.randn_like(z)
            z = z + np.sqrt(sigma_t_sq * (1 - alpha_prod_t) / (sigma_t_sq + (1 - alpha_prod_t))) * n
        return z
        
    def get_sigma_t_sq(self, denoising_step: int):
        """Get $\sigma_t^2$ for ReSample.
        
        Parameters
        ----------
        denoising_step: int
            The index of the current timestep.
            
        Returns
        -------
        torch.Tensor
            The $\sigma_t^2$ for ReSample.
        """
        alpha_prod_t = self.get_alpha_prod_t(denoising_step=denoising_step)
        alpha_prod_t_prev = (
            self.get_alpha_prod_t(denoising_step=denoising_step + 1) 
            if denoising_step < len(self.pipe.scheduler.timesteps) - 1 
            else 1
        )
        sigma_t_sq = self.options.resample_options.gamma * \
            (1 - alpha_prod_t_prev) / alpha_prod_t * \
            (1 - alpha_prod_t / alpha_prod_t_prev)
        return sigma_t_sq
    
    def estimate_t0_latent(
        self, z_t: torch.Tensor, t: torch.Tensor, noise_pred: torch.Tensor
    ):
        """Estimate the latent at time step 0, given the latent at time step t.
        
        Parameters
        ----------
        z_t: torch.Tensor
            The latent at time step t.
        t: torch.Tensor
            The timestep to estimate at.
        noise_pred: torch.Tensor
            The noise prediction at time step t.
            
        Returns
        -------
        torch.Tensor
            The estimated latent at time step 0.
        """
        alpha_prod_t = self.get_alpha_prod_t(t=t)
        z_0_hat = z_t / torch.sqrt(alpha_prod_t) - \
                  torch.sqrt(1 - alpha_prod_t) / torch.sqrt(alpha_prod_t) * noise_pred
        return z_0_hat
    
    def encode_prompt(
        self, 
        prompt: str = None, 
        negative_prompt: str = None
    ):
        """Encode the prompt into text embeddings and save them in
        `self.text_embeddings`.
        
        Parameters
        ----------
        prompt: str
            The prompt to encode.
        negative_prompt: str
            The negative prompt to encode.
        """
        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = self.pipe.encode_prompt(
            prompt=prompt,
            prompt_2=None,
            negative_prompt=negative_prompt,
            negative_prompt_2=None,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
            prompt_embeds=None,
            negative_prompt_embeds=None,
            pooled_prompt_embeds=None,
            negative_pooled_prompt_embeds=None,
            device=torch.get_default_device(),
            clip_skip=None,
            num_images_per_prompt=1,
        )
            
        self.text_embeddings.prompt_embeds = prompt_embeds
        self.text_embeddings.negative_prompt_embeds = negative_prompt_embeds
        self.text_embeddings.pooled_prompt_embeds = pooled_prompt_embeds
        self.text_embeddings.negative_pooled_prompt_embeds = negative_pooled_prompt_embeds
        
    def do_time_travel(self, i: int) -> bool:
        """Determine if time travel should be performed at timestep index `i`.
        
        Parameters
        ----------
        i: int
            The index of the current timestep.
            
        Returns
        -------
        bool
            Whether time travel should be performed.
        """
        return (
            self.options.time_travel_plan.is_enabled(i)
            and i >= self.options.time_travel_steps
            and i < len(self.pipe.scheduler.timesteps) - 1
        )
    
    def do_physical_guidance(self, i: int) -> bool:
        """Determine if physical guidance should be applied at timestep index `i`.
        
        Parameters
        ----------
        i: int
            The index of the current timestep.
            
        Returns
        -------
        bool
            Whether physical guidance should be applied.
        """
        if self.options.physical_guidance_scale <= 0:
            return False
        else:
            if (
                i < len(self.pipe.scheduler.timesteps) - 1 
                and self.options.physical_guidance_plan.is_enabled(i)
            ):
                return True
            else:
                return False

    def run_guided_sampling(self):
        # Encode the prompt.
        self.encode_prompt(self.options.prompt)
        
        # Get initial latent code.
        z = self.prepare_initial_latent()
        
        self.prepare_added_time_ids_and_embeddings()
        
        for self.current_denoise_step, t in enumerate(self.pipe.scheduler.timesteps):   
            z, z_0_hat = self.denoise_step(z, t, step_index=self.current_denoise_step)
            
            # Update the latent code with physical guidance.
            if self.do_physical_guidance(self.current_denoise_step):
                z = self.physical_guidance_step(z, z_0_hat)
            
            # Time-travel strategy
            if self.do_time_travel(self.current_denoise_step):
                # Jump back in time. `self.pipe.scheduler.timesteps` is in reverse order,
                # so we add 1 to the index to get the current timestep (it is already denoised,
                # so it is at t - 1)
                z = self.noise_step(
                    z, 
                    self.pipe.scheduler.timesteps[self.current_denoise_step + 1], 
                    by=self.options.time_travel_steps
                )
                
                # Re-denoise with physics guidance
                for j in range(self.options.time_travel_steps):
                    z, z_0_hat_tt = self.denoise_step(
                        z, 
                        self.pipe.scheduler.timesteps[self.current_denoise_step + 1 - self.options.time_travel_steps + j],
                        step_index=self.current_denoise_step + 1 - self.options.time_travel_steps + j
                    )
                    if self.options.physical_guidance_scale > 0:
                        z = self.physical_guidance_step(z, z_0_hat_tt)
            
            self.pbar.update(1)
        
        # Decode the final latents to image
        self.sampled_image = self.decode_latent(z.to(self.pipe.vae.dtype))
        self.sampled_image_pil = self.pipe.image_processor.postprocess(self.sampled_image, output_type="pil")[0]
        self.parameter_group.object.set_data(self.image_to_object(self.sampled_image))

    def run(self, n_epochs: int = None):
        n_epochs = self.options.num_epochs if n_epochs is None else n_epochs
        with torch.no_grad():
            for _ in range(n_epochs):
                self.run_pre_epoch_hooks()
                self.run_guided_sampling()


class GuidedFlowMatchingReconstructor(GuidedDiffusionReconstructor):
    """A reconstructor that works with flow matching generative models
    (Stable Diffusion 3.5, etc.).
    """
    
    pipe: StableDiffusion3Pipeline
    
    def prepare_initial_latent(self) -> torch.Tensor:
        """Get the initial latent code. The size of the latent is determined
        by the pixel-space image size and the downscaling factor of the image
        autoencoder. In general, the size is given by
        
        ```2 ** (len(self.vae.config.block_out_channels) - 1)``` 
        
        and is 8 for Stable Diffusion 3.5. The number of channels is given
        by the number of input channels of the transformer, as in
        
        ```self.pipe.transformer.config.in_channels```
        
        Returns
        -------
        torch.Tensor
            The initial latent code of shape (1, in_channels, h, w).
        """
        z = self.pipe.prepare_latents(
            batch_size=1,
            num_channels_latents=self.pipe.transformer.config.in_channels,
            height=self.parameter_group.object.lateral_shape[0],
            width=self.parameter_group.object.lateral_shape[1],
            dtype=self.text_embeddings.prompt_embeds.dtype,
            device=torch.get_default_device(),
            generator=None,
            latents=None
        )
        return z
    
    def encode_prompt(
        self, 
        prompt: str = None, 
        negative_prompt: str = None
    ):
        """Encode the prompt into text embeddings and save them in
        `self.text_embeddings`.
        
        Parameters
        ----------
        prompt: str
            The prompt to encode.
        negative_prompt: str
            The negative prompt to encode.
        """
        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = self.pipe.encode_prompt(
            prompt=prompt,
            prompt_2=None,
            prompt_3=None,
            negative_prompt=negative_prompt,
            negative_prompt_2=None,
            negative_prompt_3=None,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
            prompt_embeds=None,
            negative_prompt_embeds=None,
            pooled_prompt_embeds=None,
            negative_pooled_prompt_embeds=None,
            device=torch.get_default_device(),
            clip_skip=None,
            num_images_per_prompt=1,
            max_sequence_length=256,
            lora_scale=None,
        )
        if self.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            pooled_prompt_embeds = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)
            
        self.text_embeddings.prompt_embeds = prompt_embeds
        self.text_embeddings.negative_prompt_embeds = negative_prompt_embeds
        self.text_embeddings.pooled_prompt_embeds = pooled_prompt_embeds
        self.text_embeddings.negative_pooled_prompt_embeds = negative_pooled_prompt_embeds
        
    def decode_latent(self, z: torch.Tensor):
        """Decode the latent code to the image space and convert it to
        the structure compatible with the forward model.
        
        Parameters
        ----------
        z: torch.Tensor
            The latent code to be decoded.
            
        Returns
        -------
        torch.Tensor
            The decoded image in the structure compatible with the forward model.
        """
        z = z.to(self.pipe.vae.dtype)
        z = z / self.pipe.vae.config.scaling_factor + self.pipe.vae.config.shift_factor
        decoded_image = self.pipe.vae.decode(z, return_dict=False)[0]
        return decoded_image
    
    def encode_image(self, img: torch.Tensor):
        """Encode the image to the latent space.
        
        Parameters
        ----------
        img: torch.Tensor
            The image to be encoded.
            
        Returns
        -------
        torch.Tensor
            The encoded latent code.
        """
        img = img.to(self.pipe.vae.dtype)
        z = self.pipe.vae.encode(img).latent_dist.sample()
        z = (z - self.pipe.vae.config.shift_factor) * self.pipe.vae.config.scaling_factor
        return z
    
    def denoise_step(self, z_t: torch.Tensor, t: float):
        """Denoise the latent code by one step. Text conditioning is added
        to the noise through classifier-free guidance.
        The input is detached, so this function is not differentiable.
        
        Parameters
        ----------
        z_t: torch.Tensor
            The latent code to be denoised.
        t: float
            The timestep to denoise at.
            
        Returns
        -------
        z_tm1: torch.Tensor
            The denoised latent code (x_{t-1}).
        z_0_hat: torch.Tensor
            The estimated noise-free image at time step 0.
        """
        with torch.no_grad():
            latent_model_input = torch.cat([z_t] * 2) if self.do_classifier_free_guidance else z_t
            timestep = t.expand(latent_model_input.shape[0])
            
            # Standard denoising step.
            noise_pred = self.pipe.transformer(
                hidden_states=latent_model_input,
                timestep=timestep,
                encoder_hidden_states=self.text_embeddings.prompt_embeds,
                pooled_projections=self.text_embeddings.pooled_prompt_embeds,
                return_dict=False
            )[0]
            
            # Classifier-free (text) guidance step.
            if self.do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + self.options.text_guidance_scale * (noise_pred_text - noise_pred_uncond)
            
            step_output = self.pipe.scheduler.step(noise_pred, t, z_t, return_dict=True)
            z_tm1 = step_output.prev_sample
            if hasattr(step_output, "pred_original_sample"):
                z_0_hat = step_output.pred_original_sample
            else:
                try:
                    z_0_hat = self.estimate_t0_latent(z_tm1, t, noise_pred)
                except AttributeError:
                    logger.warning(
                        f"Unable to estimate sample at t=0 with {self.pipe.scheduler.__class__.__name__}, "
                        "so I am just using z_{t-1} as the estimate."
                    )
                    z_0_hat = z_tm1
                    
            
            if z_tm1.dtype != z_t.dtype:
                z_tm1 = z_tm1.to(z_t.dtype)
            if z_0_hat.dtype != z_t.dtype:
                z_0_hat = z_0_hat.to(z_t.dtype)
        return z_tm1, z_0_hat

    def prepare_added_time_ids_and_embeddings(self):
        return
