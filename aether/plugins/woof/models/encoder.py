import torch.nn as nn

from aether.plugins.woof.models.components import initialize_convnet_weights


class Encoder(nn.Module):
    def __init__(
        self, 
        num_in_channels: int = 1,
        num_out_channels: int = 128,
        num_levels: int = 3, 
        base_channels: int = 32, 
        use_batchnorm: bool = True,
        use_downsampling: bool = False,
        kernel_sizes_all_convs: tuple[int, int] = (3, 3),
        strides_all_convs: tuple[int, int] = (1, 1),
        paddings_all_convs: tuple[int | str, int | str] = ("same", "same"),
        dropout: float = 0.0,
    ):
        """
        Convolutional encoder model with adjustable number of levels.

        Parameters
        ----------
        num_in_channels : int
            The number of input channels.
        num_out_channels : int
            The number of output channels.
        num_levels : int
            The number of levels in the autoencoder.
        base_channels : int
            The base number of channels. The number of output channels
            of level `i` is `base_channels * 2 ** i`.
        use_batchnorm : bool
            Whether to use batch normalization.
        use_downsampling : bool
            Whether to use downsampling.
        kernel_sizes_all_convs : tuple[int, int]
            The kernel sizes of the two Conv2d layers in each downsampling block.
        strides_all_convs : tuple[int, int]
            The strides of the two Conv2d layers in each downsampling block.
        paddings_all_convs : tuple[int | str, int | str]
            The paddings of the two Conv2d layers in each downsampling block.
        dropout : float
            The dropout rate.
        """
        super().__init__()
        self.num_levels = num_levels
        self.num_in_channels = num_in_channels
        self.num_out_channels = num_out_channels
        self.base_channels = base_channels
        self.use_batchnorm = use_batchnorm
        self.use_downsampling = use_downsampling
        self.kernel_sizes_all_convs = kernel_sizes_all_convs
        self.strides_all_convs = strides_all_convs
        self.paddings_all_convs = paddings_all_convs
        self.dropout = dropout
        
        self.encoder = None
        self.build_network()
        
    def build_network(self):
        down_blocks = []
        for level in range(self.num_levels):
            down_blocks += self.get_down_block(level)
        self.encoder = nn.Sequential(
            *down_blocks
        )
        
        initialize_convnet_weights(self.encoder)
        
    def get_down_block(self, level):
        """
        Get a list of layers in a downsampling block:
        Conv2d -> BN (optional) -> ReLU -> Conv2d -> BN (optional) -> ReLU -> MaxPool2d (optional)

        Parameters
        ----------
        level : int
            0-based level index.
            
        Returns
        -------
        list[nn.Module]
            List of layers in the downsampling block.
        """
        num_in_channels = int(self.base_channels * 2 ** (level - 1))
        num_out_channels = int(self.base_channels * 2 ** level)
        if level == 0:
            num_in_channels = self.num_in_channels
        if level == self.num_levels - 1:
            num_out_channels = self.num_out_channels

        blocks = []
        blocks.append(
            nn.Conv2d(
                in_channels=num_in_channels, 
                out_channels=num_out_channels,
                kernel_size=self.kernel_sizes_all_convs[0], 
                stride=self.strides_all_convs[0], 
                padding=self.paddings_all_convs[0]
            )
        )
        if self.use_batchnorm:
            blocks.append(nn.BatchNorm2d(num_out_channels))
        blocks.append(nn.ReLU())
        blocks.append(
            nn.Conv2d(
                in_channels=num_out_channels, 
                out_channels=num_out_channels, 
                kernel_size=self.kernel_sizes_all_convs[1], 
                stride=self.strides_all_convs[1], 
                padding=self.paddings_all_convs[1]
            )
        )
        if self.use_batchnorm:
            blocks.append(nn.BatchNorm2d(num_out_channels))
        blocks.append(nn.ReLU())
        if self.dropout > 0.0:
            blocks.append(nn.Dropout(self.dropout))
        if self.use_downsampling:
            blocks.append(nn.MaxPool2d((2, 2)))

        return blocks

    def forward(self, x):
        """
        Forward pass through the encoder.

        Parameters
        ----------
        x: torch.Tensor
            A (n, num_input_channels, h, w) tensor.

        Returns
        -------
        torch.Tensor
            A (n, num_output_channels, h, w) tensor.
        """
        return self.encoder(x)
