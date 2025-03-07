from typing import Literal
import logging

from PIL import Image
import torch
from diffusers import StableDiffusion3Pipeline, DDIMScheduler
import diffusers.utils as dutils
from ptychi.reconstructors.ad_ptychography import AutodiffPtychographyReconstructor
from ptychi.data_structures.parameter_group import PtychographyParameterGroup
import ptychi.maps as maps
from ptychi.io_handles import PtychographyDataset
import firefly.api as api
from firefly.io import HuggingFaceStableDiffusionModelLoader


logger = logging.getLogger(__name__)


class GuidedDiffusionReconstructor(AutodiffPtychographyReconstructor):
    
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
        
        self.model_loader = model_loader
        self.pipe: StableDiffusion3Pipeline = None
        self.loss_function = maps.get_loss_function_by_enum(options.loss_function)()
        self.forward_model = None
                
        self.text_embeddings = {}
        self.sampled_image: Image.Image = None
        self._do_classifier_free_guidance = self.options.text_guidance_scale > 1
        
    def build(self):
        self.build_pipe()
        super().build()
        
    def build_pipe(self):
        self.model_loader.load()
        self.pipe = self.model_loader.pipe
        
        # self.pipe.scheduler = DDIMScheduler.from_config(self.pipe.scheduler.config)
        
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
        
    def noise_step(self, z_t: torch.Tensor, t: int, by: int):
        """Add noise to the latent code.
        
        Parameters
        ----------
        z_t: torch.Tensor
            The latent code to be denoised.
        t: int
            The timestep to denoise at.
        by: int
            The number of steps to add noise by.
            
        Returns
        -------
        torch.Tensor
            The latent code after adding noise.
        """
        # Get the noise schedule for the current timestep
        alpha_prod_t = self.pipe.scheduler.alphas_cumprod[t]
        alpha_prod_t_prev = (
            self.pipe.scheduler.alphas_cumprod[t + by] 
            if t + by < len(self.pipe.scheduler.alphas_cumprod) 
            else torch.tensor(0.0)
        )
        
        # Sample random noise
        noise = torch.randn_like(z_t)
        
        # Calculate the noisy sample x_{t+n} from z_t using the noise schedule
        z_t_plus_n = torch.sqrt(alpha_prod_t_prev / alpha_prod_t) * z_t + \
                     torch.sqrt(1 - alpha_prod_t_prev / alpha_prod_t) * noise
        
        return z_t_plus_n
        
    def denoise_step(self, z_t: torch.Tensor, t: int):
        """Denoise the latent code by one step. Text conditioning is added
        to the noise through classifier-free guidance.
        The input is detached, so this function is not differentiable.
        
        Parameters
        ----------
        z_t: torch.Tensor
            The latent code to be denoised.
        t: int
            The timestep to denoise at.
            
        Returns
        -------
        z_tm1: torch.Tensor
            The denoised latent code (x_{t-1}).
        z_0_hat: torch.Tensor
            The estimated noise-free image at time step 0.
        """
        with torch.no_grad():
            latent_model_input = torch.cat([z_t] * 2) if self._do_classifier_free_guidance else z_t
            timestep = t.expand(latent_model_input.shape[0])
            
            # Standard denoising step.
            noise_pred = self.pipe.transformer(
                hidden_states=latent_model_input,
                timestep=timestep,
                encoder_hidden_states=self.text_embeddings["prompt_embeds"],
                pooled_projections=self.text_embeddings["pooled_prompt_embeds"],
                return_dict=False
            )[0]
            
            # Classifier-free (text) guidance step.
            if self._do_classifier_free_guidance:
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
    
    def physical_guidance_step(self, z_t: torch.Tensor, z_0_hat: torch.Tensor):
        """Update the latent code using the physical guidance.
        
        Parameters
        ----------
        z_t: torch.Tensor
            The latent code at time step t.
        z_0_hat: torch.Tensor
            The estimated noise-free image at time step 0.
        """
        score = self.calculate_physical_guidance_score(z_0_hat)
        z_t = z_t - self.options.physical_guidance_scale * score
        return z_t
        
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
        z = self.pipe.vae.encode(img).latent_dist.sample()
        z = (z - self.pipe.vae.config.shift_factor) * self.pipe.vae.config.scaling_factor
        return z
    
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
        obj = obj.to(torch.complex64)
        return obj
    
    def object_to_image(
        self, 
        obj: torch.Tensor, 
        representation: Literal["real_imag", "mag_phase"] = "mag_phase"
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
            # TODO: allow magnitude to be also generated
            # TODO: phase wrapping
            phase = obj.angle()
            img = phase.unsqueeze(1)
            img = img.repeat(1, 3, 1, 1)
        else:
            raise ValueError(f"Invalid representation: {representation}")
        img = img.to(self.pipe.dtype)
        return img
        
    def calculate_physical_guidance_score(self, z_t: torch.Tensor):
        """Denoise the latent code by one step, decode it, and calculate
        the score function of the physical model $f$, given by
        $\nabla_{z_{t-1}} || f(D(z_{t-1})) - y ||^2$,  where $D$ is the
        decoder and $f$ is the forward model.
        
        Parameters
        ----------
        z_t: torch.Tensor
            A (1, in_channels, h, w) tensor giving the latent code at 
            timestep t.
            
        Returns
        -------
        torch.Tensor
            The score function of the physical model.
        """
        if z_t.shape[0] != 1:
            raise ValueError("The length of the batch dimension of z_t must be 1.")
        
        decoded_image = self.decode_latent(z_t)
        o_hat = self.image_to_object(decoded_image)
        
        with torch.no_grad():
            # TODO: convert z_t to the correct format.
            self.forward_model.object.set_data(o_hat)
        
        accumulated_grad = torch.zeros_like(o_hat)
        with torch.enable_grad():
            for batch_data in self.dataloader:
                self.forward_model.object.initialize_grad()
                input_data, y_true = self.prepare_batch_data(batch_data)
                y_pred = self.forward_model(*input_data)
                batch_loss = self.loss_function(
                    y_pred[:, self.dataset.valid_pixel_mask], y_true[:, self.dataset.valid_pixel_mask]
                )
                batch_loss.backward()
                accumulated_grad += self.forward_model.object.get_grad() / len(input_data[0])
        
        accumulated_grad = self.object_to_image(accumulated_grad)
        score = self.encode_image(accumulated_grad)
        return score
    
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
        alpha_prod_t = self.pipe.scheduler.alphas_cumprod[t]
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
            prompt_3=None,
            negative_prompt=negative_prompt,
            negative_prompt_2=None,
            negative_prompt_3=None,
            do_classifier_free_guidance=self._do_classifier_free_guidance,
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
        if self._do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            pooled_prompt_embeds = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)
            
        self.text_embeddings["prompt_embeds"] = prompt_embeds
        self.text_embeddings["negative_prompt_embeds"] = negative_prompt_embeds
        self.text_embeddings["pooled_prompt_embeds"] = pooled_prompt_embeds
        self.text_embeddings["negative_pooled_prompt_embeds"] = negative_pooled_prompt_embeds
        
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
            i % self.options.time_travel_interval == 0 
            and i >= self.options.time_travel_steps
        )
        
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
            dtype=self.text_embeddings["prompt_embeds"].dtype,
            device=torch.get_default_device(),
            generator=None,
            latents=None
        )
        return z

    def run_guided_sampling(self):
        # Encode the prompt.
        self.encode_prompt(self.options.prompt)
        
        # Get initial latent code.
        z = self.prepare_initial_latent()
        
        # Set timesteps.
        self.pipe.scheduler.set_timesteps(self.options.num_inference_steps)
        
        for i, t in enumerate(self.pipe.scheduler.timesteps):
            z, z_0_hat = self.denoise_step(z, t)
            
            # Update the latent code with physical guidance.
            if self.options.physical_guidance_scale > 0:
                z = self.physical_guidance_step(z, z_0_hat)
            
            # Time-travel strategy
            if self.do_time_travel(i):
                # Jump back in time
                z = self.noise_step(z, t, by=self.options.time_travel_steps)
                
                # Re-denoise with physics guidance
                for j in range(self.options.time_travel_steps):
                    z, z_0_hat_tt = self.denoise_step(z, t + self.options.time_travel_steps - j)
                    if self.options.physical_guidance_scale > 0:
                        z = self.physical_guidance_step(z, z_0_hat_tt)
            
            self.pbar.update(1)
        
        # Decode the final latents to image
        decoded_image = self.decode_latent(z)
        self.sampled_image = self.pipe.image_processor.postprocess(decoded_image, output_type="pil")[0]

    def run(self, n_epochs: int = None):
        n_epochs = self.options.num_epochs if n_epochs is None else n_epochs
        with torch.no_grad():
            for _ in range(n_epochs):
                self.run_guided_sampling()
