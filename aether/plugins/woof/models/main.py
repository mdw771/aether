from typing import Optional

import torch
import torch.nn as nn


class FocalStackMaskPredictor(nn.Module):
    def __init__(
        self,
        feature_extractor: nn.Module,
        latent_processor: nn.Module,
        decoder: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.latent_processor = latent_processor
        self.decoder = decoder
        
        if self.latent_processor.num_in_channels != self.feature_extractor.num_out_channels:
            raise ValueError(
                "The number of input channels of the latent processor must be the same as the "
                "number of output channels of the feature extractor."
            )
        
        if self.latent_processor.num_projected_out_dims is not None and self.decoder is not None:
            if decoder.num_in_channels != self.latent_processor.num_projected_out_dims:
                raise ValueError(
                    f"The number of input channels of the decoder is {decoder.num_in_channels} "
                    f"but the transformer outputs {self.latent_processor.num_projected_out_dims} "
                    f"channels."
                )
        else:
            if decoder.num_in_channels != self.latent_processor.num_hidden_dims:
                raise ValueError(
                    f"The number of input channels of the decoder is {decoder.num_in_channels} "
                    f"but the transformer outputs {self.latent_processor.num_hidden_dims} "
                    f"channels."
                )

    def forward(self, x):
        """Forward pass through the model.
        
        Parameters
        ----------
        x: torch.Tensor
            A (n_slices, num_input_channels, h, w) tensor.
            
        Returns
        -------
        torch.Tensor
            A (n_slices, 3, h, w) tensor giving the predicted masks
            for each slice in the stack. The 3 channels respectively
            give the probabilities of in-focus, out-of-focus, and background.
        """
        n_slices = x.shape[0]
        h_in, w_in = x.shape[2:]
        
        # Output shape: (n_slices, num_feature_extractor_output_channels, h, w)
        x = self.feature_extractor(x)
        output_shape_feature_extractor = x.shape
        
        # Permute and reshape to (h * w, n_slices, num_feature_extractor_output_channels)
        x = x.permute(2, 3, 0, 1)
        x = x.reshape(-1, n_slices, x.shape[-1])
        
        # Output shape: (h * w, n_slices, num_projected_out_dims)
        x = self.latent_processor(x)
        
        # Permute and reshape to (n_slices, num_projected_out_dims, h_encoder, w_encoder)
        x = x.permute(1, 2, 0)
        x = x.reshape(n_slices, x.shape[1], output_shape_feature_extractor[-2], output_shape_feature_extractor[-1])
        
        if self.decoder is not None:
            x = self.decoder(x, (h_in, w_in))
        return x


class FocalStackOOFCleaner(nn.Module):
    def __init__(
        self,
        feature_extractor: nn.Module,
        latent_processor: nn.Module,
        decoder: nn.Module,
    ):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.latent_processor = latent_processor
        self.decoder = decoder
                
        if self.latent_processor.num_projected_out_dims is not None and self.decoder is not None:
            if decoder.num_in_channels != self.latent_processor.num_projected_out_dims:
                raise ValueError(
                    f"The number of input channels of the decoder is {decoder.num_in_channels} "
                    f"but the transformer outputs {self.latent_processor.num_projected_out_dims} "
                    f"channels."
                )
                
    def forward(self, x: torch.Tensor):
        """Forward pass through the model.
        
        Parameters
        ----------
        x: torch.Tensor
            A (n_slices, n_channels, h, w) tensor giving the input image stack.
            
        Returns
        -------
        torch.Tensor
            A tensor with the same shape as the input.
        """
        # Output shape: (n_slices, num_features, h', w')
        x = self.feature_extractor(x)
        feature_extractor_output_shape = x.shape
        
        # Output shape: (h' * w', n_slices, num_features)
        x = x.permute(2, 3, 0, 1)
        x = x.view(-1, x.shape[-2], x.shape[-1])
        
        # Output shape: (h' * w', n_slices, latent_processor.num_projected_out_dims)
        x = self.latent_processor(x)
        
        # Output shape: (n_slices, latent_processor.num_projected_out_dims, h', w')
        x = x.permute(1, 2, 0).view(
            x.shape[1], 
            x.shape[2], 
            feature_extractor_output_shape[-2], 
            feature_extractor_output_shape[-1]
        )
        
        # Output shape: (n_slices, num_out_channels, h, w)
        x = self.decoder(x)
            
        return x