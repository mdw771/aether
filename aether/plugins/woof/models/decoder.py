from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from aether.plugins.woof.models.components import initialize_convnet_weights


class Decoder(nn.Module):
    def __init__(
        self, 
        num_in_channels: int = 128, 
        num_out_channels: int = 1,
        num_levels: int = 3, 
        base_channels: int = 32, 
        dropout: float = 0.0,
        use_batchnorm: bool = False,
        use_interpolation_in_last_layer: bool = False,
        use_skip_connection: bool = False,
    ):
        """
        Convolutional decoder model with adjustable number of levels.

        Parameters
        ----------
        num_in_channels : int
            The number of input channels.
        num_levels : int
            The number of levels in the autoencoder.
        base_channels : int
            The base number of channels. In the encoder part, the number of output channels
            of level `i` is `base_channels * 2 ** i`.
        dropout : float
            The dropout rate.
        use_batchnorm : bool
            Whether to use batch normalization.
        use_interpolation_in_last_layer : bool
            Whether to use interpolation in the last layer. WHen this is True, `target_size`
            must be given when calling the forward method.
        use_skip_connection : bool
            If True, the forward method will take an additional argument `skip_conn_input`
            with the same size as the output tensor, which will be added to the output.
        """
        super().__init__()
        self.num_levels = num_levels
        self.num_in_channels = num_in_channels
        self.num_out_channels = num_out_channels
        self.base_channels = base_channels
        self.dropout = dropout
        self.use_batchnorm = use_batchnorm
        self.use_interpolation_in_last_layer = use_interpolation_in_last_layer
        self.use_skip_connection = use_skip_connection
        
        self.decoder_main = None
        self.decoder_last_layer = None
        self.build_network()

    def create_generic_decoder_layers(self) -> list[nn.Module]:
        decoder_layers = []
        for level in range(self.num_levels - 1, -1, -1):
            decoder_layers += self.get_up_block(level)
        return decoder_layers
        
    def build_network(self):
        self.decoder_main = nn.Sequential(*self.create_generic_decoder_layers())
        
        decoder_last_layer = [nn.Conv2d(self.base_channels, 1, 3, stride=1, padding=(1, 1))]
        self.decoder_last_layer = nn.Sequential(*decoder_last_layer)
        
        initialize_convnet_weights(self.decoder_main)
        initialize_convnet_weights(self.decoder_last_layer)

    def get_up_block(self, level):
        """
        Get a list of layers in a upsampling block:
        Conv2d -> BN (optional) -> ReLU -> Conv2d -> BN (optional) -> ReLU -> Upsample (scale_factor=2, mode="bilinear")

        Parameters
        ----------
        level : int
            0-based level index.
            
        Returns
        -------
        list[nn.Module]
            List of layers in the upsampling block.
        """
        if level == self.num_levels - 1:
            num_in_channels = self.num_in_channels
            num_out_channels = self.base_channels * 2 ** level
        else:
            num_in_channels = self.base_channels * 2 ** (level + 1)
            num_out_channels = self.base_channels * 2 ** level
        num_in_channels = int(num_in_channels)
        num_out_channels = int(num_out_channels)

        blocks = []
        blocks.append(nn.Conv2d(num_in_channels, num_out_channels, 3, stride=1, padding=(1, 1)))
        if self.use_batchnorm:
            blocks.append(nn.BatchNorm2d(num_out_channels))
        blocks.append(nn.ReLU())
        blocks.append(nn.Conv2d(num_out_channels, num_out_channels, 3, stride=1, padding=(1, 1)))
        if self.use_batchnorm:
            blocks.append(nn.BatchNorm2d(num_out_channels))
        blocks.append(nn.ReLU())
        if self.dropout > 0.0:
            blocks.append(nn.Dropout(self.dropout))
        if not (level == self.num_levels - 1 and self.use_interpolation_in_last_layer):
            blocks.append(nn.Upsample(scale_factor=2, mode="bilinear"))
        return blocks

    def forward(
        self, 
        x: torch.Tensor, 
        target_size: Optional[tuple[int, int]] = None, 
        skip_conn_input: Optional[torch.Tensor] = None,
        return_tensor_before_skip_conn: bool = False
    ):
        """
        Forward pass through the decoder.

        Parameters
        ----------
        x: torch.Tensor
            A (n, num_input_channels, h, w) tensor.
        target_size: Optional[tuple[int, int]]
            The target size of the output. If `use_interpolation_in_last_layer`
            is True, this must be given.
        skip_conn_input: Optional[torch.Tensor]
            A (n, num_input_channels, h, w) tensor. If `use_skip_connection`
            is True, this must be given.
        return_tensor_before_skip_conn: bool
            If True, the tensor before the skip connection will be returned
            along with the output tensor.

        Returns
        -------
        torch.Tensor | tuple[torch.Tensor, torch.Tensor]
            If `return_tensor_before_skip_conn` is False, a (n, num_output_channels, h, w) tensor.
            If `return_tensor_before_skip_conn` is True, a tuple of two (n, num_output_channels, h, w) tensors.
        """
        x = self.decoder_main(x)
        if self.use_interpolation_in_last_layer:
            if target_size is None:
                raise ValueError("target_size must be given when use_interpolation_in_last_layer is True.")
            x = F.interpolate(x, size=target_size, mode="bilinear")
        x = self.decoder_last_layer(x)
        
        x_before_skip_conn = x
        if self.use_skip_connection:
            if skip_conn_input is None:
                raise ValueError("skip_conn_input must be given when use_skip_connection is True.")
            x = x + skip_conn_input
        if return_tensor_before_skip_conn:
            return x, x_before_skip_conn
        else:
            return x
    
    
class UpsampleConv2d(nn.Module):
    def __init__(self, num_in_channels: int, num_out_channels: int):
        super().__init__()
        self.num_in_channels = num_in_channels
        self.num_out_channels = num_out_channels
        
        self.conv = nn.Conv2d(num_in_channels, num_out_channels, 3, stride=1, padding=(1, 1))

    def forward(self, x, target_size: tuple[int, int]):
        x = F.interpolate(x, size=target_size, mode="bilinear")
        x = self.conv(x)
        return x


class PointwiseDecoder(nn.Module):
    def __init__(self, num_in_channels: int, num_out_channels: int):
        super().__init__()
        self.num_in_channels = num_in_channels
        self.num_out_channels = num_out_channels
        
        self.projector = nn.Linear(num_in_channels, num_out_channels)
        
    def forward(self, x):
        x = self.projector(x)
        return x
