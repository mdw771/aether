from typing import Optional
import logging

import torch
import ptychi.maps as pcmaps

import aether.image_proc as fip
from aether.reconstructors.guided_sampling import GuidedLatentDiffusionReconstructor


logger = logging.getLogger(__name__)


class ADMMDiffReconstructor(GuidedLatentDiffusionReconstructor):
    
    def physical_guidance_step(
        self, 
        z_t: torch.Tensor, 
        z_0_hat: torch.Tensor, 
        t: Optional[torch.Tensor] = None, 
        noise_pred: Optional[torch.Tensor] = None
    ):
        """Update the latent code using the physical guidance.
        
        Parameters
        ----------
        z_t: torch.Tensor
            The latent code at time step t.
        z_0_hat: torch.Tensor
            The estimated noise-free image at time step 0.
        t: torch.Tensor
            The timestep.
        noise_pred: torch.Tensor
            The predicted noise at time step t.
            
        Returns
        -------
        torch.Tensor
            The updated latent code.
        """
        z_0_hat = z_0_hat.to(torch.float32)
        z_t = z_t.to(torch.float32)

        z_opt = self.calculate_physical_guidance_optimization(
            z_t,
            optimize_against_latent=True,
            t=t,
            noise_pred=noise_pred,
            estimate_t0_in_optimization_loop=True
        )
        return z_opt.to(self.pipe.unet.dtype)
        
    def calculate_physical_guidance_optimization(
        self, 
        z: torch.Tensor, 
        optimize_against_latent: bool = False,
        estimate_t0_in_optimization_loop: bool = False,
        t: Optional[torch.Tensor] = None,
        noise_pred: Optional[torch.Tensor] = None
    ):
        """Optimize the image or latent to minimize the physics negative log-likelihood.
        
        Parameters
        ----------
        z: torch.Tensor
            The latent to optimize.
        optimize_against_latent: bool
            If True, optimization is done directly against the latent code, 
            with backpropagation running through the decoder. Otherwise, the
            image is decoded, and the optimization is done against the image,
            with backpropagation running through the encoder.
        estimate_t0_in_optimization_loop: bool
            If True, the given latent is assumed to be at time step t, and
            the latent at time step 0 is estimated using Tweedie's formula.
            The estimation step will be included in the optimization loop
            so backpropagation will run through Tweedie's formula.
            This argument is ignored if `optimize_against_latent` is False.
        t: torch.Tensor
            The timestep. Only used if `estimate_t0_in_optimization_loop` is True.
        noise_pred: torch.Tensor
            The predicted noise at time step t. Only used if 
            `estimate_t0_in_optimization_loop` is True.
            
        Returns
        -------
        torch.Tensor
            The resampled latent code.
        """
        if z.shape[0] != 1:
            raise ValueError("The length of the batch dimension of z_t must be 1.")

        if not optimize_against_latent:
            x = self.decode_latent(z)

        if self.use_admm:
            if optimize_against_latent:
                z0 = z.detach().clone()
            else:
                x0 = x.detach().clone()

        optimizer = pcmaps.get_optimizer_by_enum(
            self.options.resample_options.optimizer
        )(
            [z if optimize_against_latent else x], lr=self.options.resample_options.step_size
        )

        with torch.enable_grad():
            if not optimize_against_latent:
                x = x.requires_grad_(True)
            else:
                z = z.requires_grad_(True)

            for i in range(self.options.resample_options.num_z_optimization_epochs):
                for batch_data in self.dataloader:
                    # self.forward_model.zero_grad() does not zero gradients of x.
                    if not optimize_against_latent:
                        x.grad = None
                        decoded_image = x
                    else:
                        z.grad = None
                        if estimate_t0_in_optimization_loop:
                            z_hat_0 = self.estimate_t0_latent(z, t, noise_pred)
                        else:
                            z_hat_0 = z
                        decoded_image = self.decode_latent(z_hat_0)

                    o_hat = fip.image_to_object(img_mag=1, img_phase=decoded_image)
                    self.set_object_data_to_forward_model(o_hat)

                    input_data, y_true = self.prepare_batch_data(batch_data)
                    y_pred = self.forward_model(*input_data)
                    batch_loss = self.loss_function(
                        y_pred[:, self.dataset.valid_pixel_mask], y_true[:, self.dataset.valid_pixel_mask]
                    )

                    # Add frequency loss term.
                    if self.options.resample_options.frequency_loss_weight > 0:
                        frequency_loss = self.frequency_loss(o_hat)
                        batch_loss = batch_loss + self.options.resample_options.frequency_loss_weight * frequency_loss

                    # Add ADMM proximal penalty term.
                    if self.use_admm:
                        if optimize_against_latent:
                            prox = self.options.proximal_penalty / 2 * (
                                z - z0
                            ).norm() ** 2
                        else:
                            prox = self.options.proximal_penalty / 2 * (
                                x - x0
                            ).norm() ** 2
                        batch_loss = batch_loss + prox

                    batch_loss.backward(retain_graph=True)
                    optimizer.step()
                    self.step_all_optimizers()
                    self.forward_model.zero_grad()
                    self.loss_tracker.update_batch_loss_with_value(batch_loss.item())
                self.loss_tracker.conclude_epoch()
                self.loss_tracker.print_latest()

        if not optimize_against_latent:
            x = x.detach()
            z = self.encode_image(x)
        else:
            z = z.detach()
        return z
    
    def run_guided_sampling(self):
        # Encode the prompt.
        self.encode_prompt(self.options.prompt)

        # Get initial latent code.
        self.z = self.prepare_initial_latent()
        self.create_admm_variables(self.z)

        self.prepare_added_time_ids_and_embeddings()

        for self.current_denoise_step, t in enumerate(self.pipe.scheduler.timesteps):
            # z-step
            denoise_input = self.p - self.v if (self.use_admm and self.current_denoise_step > 0) else self.z
            self.z, z_0_hat = self.denoise_step(denoise_input, t, step_index=self.current_denoise_step)

            # p-step: update the latent code with physical guidance.
            if self.do_physical_guidance(self.current_denoise_step):
                if self.use_admm:
                    physical_guidance_input = self.z + self.v
                    _, z_0_hat, noise_pred = self.denoise_step(
                        physical_guidance_input, t, step_index=self.current_denoise_step, return_noise_pred=True
                    )
                else:
                    physical_guidance_input = self.z
                self.p = self.physical_guidance_step(
                    physical_guidance_input, 
                    z_0_hat,
                    t=t,
                    noise_pred=noise_pred
                )
                if not self.use_admm:
                    self.z = self.p
            elif self.use_admm:
                self.p = self.z + self.v

            # Time-travel strategy
            if self.do_time_travel(self.current_denoise_step):
                # Jump back in time. `self.pipe.scheduler.timesteps` is in reverse order,
                # so we add 1 to the index to get the current timestep (it is already denoised,
                # so it is at t - 1)
                self.z = self.noise_step(
                    self.z, 
                    self.pipe.scheduler.timesteps[self.current_denoise_step + 1], 
                    by=self.options.time_travel_steps
                )

                # Re-denoise with physics guidance
                for j in range(self.options.time_travel_steps):
                    self.z, z_0_hat_tt = self.denoise_step(
                        self.z, 
                        self.pipe.scheduler.timesteps[self.current_denoise_step + 1 - self.options.time_travel_steps + j],
                        step_index=self.current_denoise_step + 1 - self.options.time_travel_steps + j
                    )
                    if self.options.physical_guidance_scale > 0:
                        self.z = self.physical_guidance_step(self.z, z_0_hat_tt)

            self.update_dual()

            self.pbar.update(1)

        # Decode the final latents to image
        self.sampled_image = self.decode_latent(self.z.to(self.pipe.vae.dtype))
        self.sampled_image_pil = self.pipe.image_processor.postprocess(self.sampled_image, output_type="pil")[0]
        self.parameter_group.object.set_data(fip.image_to_object(img_mag=1, img_phase=self.sampled_image))
        self.parameter_group.object.set_data(fip.image_to_object(img_mag=1, img_phase=self.sampled_image))
