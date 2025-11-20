# Copyright © 2025 UChicago Argonne, LLC All right reserved
# Full license accessible at https://github.com//AdvancedPhotonSource/aether/blob/main/LICENSE

import yaml
import os

import torch
from torch import Tensor

import aether
from aether.reconstructors.pnp import PnPImageEditingReconstructor
import aether.plugins.woof as woof
import aether.api as api
from aether.image_proc import ImageStandardizer, pad_to_divisible_by_patch_size


class PnPWoofReconstructor(PnPImageEditingReconstructor):
    model_config: dict = None
        
    def build(self):
        super().build()
        self.build_model()
        
    @staticmethod
    def get_model_config_without_classname(config: dict) -> dict:
        return {k: v for k, v in config.items() if k != "model_class"}
    
    def load_checkpoint(self):
        checkpoint = torch.load(self.options.prior_projection_options.checkpoint_path)
        self.model.load_state_dict(checkpoint["model"])
    
    def build_model(self):        
        with open(self.options.prior_projection_options.config_path, "r") as f:
            self.model_config = yaml.safe_load(f)
        
        encoder = getattr(
            woof.models.encoder, self.model_config["feature_extractor_config"]["model_class"]
        )(
            **self.get_model_config_without_classname(self.model_config["feature_extractor_config"])
        )
        latent_processor = getattr(
            woof.models.latent_processor, self.model_config["latent_processor_config"]["model_class"]
        )(
            **self.get_model_config_without_classname(self.model_config["latent_processor_config"])
        )
        decoder = getattr(
            woof.models.decoder, self.model_config["decoder_config"]["model_class"]
        )(
            **self.get_model_config_without_classname(self.model_config["decoder_config"])
        )
        self.model = woof.models.main.FocalStackOOFCleaner(encoder, latent_processor, decoder)
        
        self.load_checkpoint()
        
        self.model.to(torch.get_default_device())

    def run_editing(self, orig_img_mag: Tensor, orig_img_phase: Tensor):
        """Run image editing.
        
        Parameters
        ----------
        orig_img_phase: torch.Tensor
            A (n_slices, 3, h, w) tensor giving the original phase image.
        orig_img_mag: torch.Tensor
            A (n_slices, 3, h, w) tensor giving the original magnitude image.
            
        Returns
        -------
        edited_mag_imgs: list[torch.Tensor]
            A list of (n_slices, 3, h, w) tensors giving the edited magnitude images.
        edited_phase_imgs: list[torch.Tensor]
            A list of (n_slices, 3, h, w) tensors giving the edited phase images.
        """
        assert isinstance(self.options.prior_projection_options, api.WoofOptions)
        
        n_input_channels = orig_img_mag.shape[1]
        
        edited_image_components = []
        for i, orig_img_component in enumerate([orig_img_mag, orig_img_phase]):
            if (
                not self.options.prior_projection_options.edit_magnitude
                and i == 0
            ):
                edited_image_components.append(orig_img_component)
                if self.options.prior_projection_options.constant_magnitude_value is not None:
                    edited_image_components.append(self.options.prior_projection_options.constant_magnitude_value)
                continue
            
            if (
                not self.options.prior_projection_options.edit_phase 
                and i == 1
            ):
                edited_image_components.append(orig_img_component)
                if self.options.prior_projection_options.constant_phase_value is not None:
                    edited_image_components.append(self.options.prior_projection_options.constant_phase_value)
                continue
        
            # Average over channel dimension since model expects a single channel input.
            orig_img_component = orig_img_component.mean(dim=1, keepdim=True)

            # Standardize and pad the image to be divisible by the patch size
            standardizer = ImageStandardizer()
            orig_img_component = standardizer.standardize(orig_img_component)
            orig_img_component, (pad_y, pad_x) = pad_to_divisible_by_patch_size(
                orig_img_component,
                patch_size=self.model_config["training_config"]["patch_size"]
            )
            
            # Run the model
            edited_img_component = self.model(orig_img_component)
            
            # Unstandardize and unpad the image
            if pad_y > 0:
                edited_img_component = edited_img_component[:, :, :-pad_y, :]
            if pad_x > 0:
                edited_img_component = edited_img_component[:, :, :, :-pad_x]
            edited_img_component = standardizer.unstandardize(edited_img_component)
            
            edited_image_components.append(edited_img_component.repeat(1, n_input_channels, 1, 1))
            
        edited_image_mag, edited_image_phase = edited_image_components
        return edited_image_mag, edited_image_phase
