from typing import Optional
import logging
import inspect
import torch
import ptychi.maps as pcmaps

import aether.image_proc as fip
from aether.reconstructors.guided_sampling import GuidedLatentDiffusionReconstructor


logger = logging.getLogger(__name__)


class LatentDPSReconstructor(GuidedLatentDiffusionReconstructor):
    
    def denoise_step(self, z_t: torch.Tensor, t: int, step_index: Optional[int] = None, return_noise_pred: bool = False):
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
        return_noise_pred: bool
            If True, the predicted noise at time step t will be returned.
            
        Returns
        -------
        z_tm1: torch.Tensor
            The denoised latent code (x_{t-1}).
        z_0_hat: torch.Tensor
            The estimated noise-free image at time step 0.
        noise_pred: torch.Tensor
            The predicted noise at time step t. Only returned if
            `return_noise_pred` is True.
        """
        z_tm1, _, noise_pred = super().denoise_step(z_t, t, step_index, return_noise_pred=True)
        
        # Use DPS paper's formula to calculate z_0_hat.
        z_0_hat = 1 / torch.sqrt(self.get_alpha_prod_t(t=t)) * (z_t + (1 - self.get_alpha_prod_t(t=t)) * noise_pred)
        
        if return_noise_pred:
            return z_tm1, z_0_hat, noise_pred
        else:
            return z_tm1, z_0_hat
    
    def physical_guidance_step(self, z_t: torch.Tensor, z_0_hat: torch.Tensor):
        """Update the latent code using the physical guidance.
        
        Parameters
        ----------
        z_t: torch.Tensor
            The latent code at time step t.
        z_0_hat: torch.Tensor
            The estimated noise-free image at time step 0.
            
        Returns
        -------
        torch.Tensor
            The updated latent code.
        """
        z_0_hat = z_0_hat.to(torch.float32)
        z_t = z_t.to(torch.float32)

        score = self.calculate_physical_guidance_score(z_0_hat)
        score = self.remove_score_function_outliers(score)
        z_t = z_t - self.options.physical_guidance_scale * score
        return z_t.to(self.pipe.unet.dtype)
